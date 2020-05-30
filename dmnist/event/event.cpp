#include <torch/torch.h>
#include <iostream>
#include "mpi.h"

float findThreshold(int, float, float);

std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

// Define a new Module.
struct Model : torch::nn::Module {
    Model()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 128));
        fc2 = register_module("fc2", torch::nn::Linear(128, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::relu(fc2->forward(x));
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main(int argc, char *argv[])
{
    // parsing runtime args
    // float constant = (float)std::atof(argv[1]);
    // float gamma = (float)std::atof(argv[2]);
    float parameter = (float)std::atof(argv[1]);

    auto D_in = 784;  // dimension of input layer
    auto H = 128;     // dimension of hidden layer
    auto D_out = 10;  // dimension of output layer

    // history at sender and receiver
    auto sent_history = 2;
    auto recv_history = 2;

    // MPI variables
    int rank, numranks;
    int left, right;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    MPI_Request req1, req2;
    MPI_Win win;

    // ring arrangement
    if (rank == 0)
        left = numranks - 1;
    else
        left = rank - 1;

    if (rank == numranks - 1)
        right = 0;
    else
        right = rank + 1;

    // end MPI variables

    // Timer variables
    auto tstart = 0.0;
    auto tend = 0.0;

    // Read dataset
    // std::string filename = "../../../mnist/data";
    std::string filename =
        "/afs/crc.nd.edu/user/s/sghosh2/Public/ML/mnist/data";
    auto dataset =
        torch::data::datasets::MNIST(filename)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    // Distributed Random Sampler - use sequential for just gradient descent
    auto data_sampler = torch::data::samplers::DistributedSequentialSampler(
        dataset.size().value(), numranks, rank, false);

    auto num_train_samples_per_pe = dataset.size().value() / numranks;

    // Generate dataloader
    auto batch_size = num_train_samples_per_pe;
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     data_sampler, batch_size);

    // sets manual seed - randomness in sgd later ?
    torch::manual_seed(0);

    auto model = std::make_shared<Model>();

    auto sz = model->named_parameters().size();
    auto param = model->named_parameters();
    auto num_elem_param = (H * D_in + H) + (D_out * H + D_out);
    auto param_elem_size = param[0].value().element_size();

    // create memory window for parameters from left and right (STILL CASTING AS
    // FLOAT TYPE)
    auto win_size = 2 * num_elem_param;
    // auto win_mem = (float*)calloc(win_size, win_size*param_elem_size);
    float *win_mem;
    MPI_Alloc_mem(win_size * param_elem_size, MPI_INFO_NULL, &win_mem);
    MPI_Win_create(win_mem, win_size * param_elem_size, param_elem_size,
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // initializing RMA window
    for (int i = 0; i < win_size; i++) {
        *(win_mem + i) = 0.0;
    }

    // Threshold
    float thres[sz];

    // history of values at the sender - to estimate the threshold
    float last_sent_values_norm[sz];
    float last_sent_iters[sz];
    float sent_slopes_norm[sz][sent_history];

    // history of values at the receiver - to extrapolate parameters between new
    // messages
    float left_last_recv_values[num_elem_param];
    float left_last_recv_values_norm[sz];
    float left_last_recv_iters[sz];
    float left_recv_slopes[num_elem_param][recv_history];

    float right_last_recv_values[num_elem_param];
    float right_last_recv_values_norm[sz];
    float right_last_recv_iters[sz];
    float right_recv_slopes[num_elem_param][recv_history];

    // initializing values
    for (int i = 0; i < sz; i++) {
        last_sent_values_norm[i] = 0.0;
        last_sent_iters[i] = 0.0;

        left_last_recv_values_norm[i] = 0.0;
        right_last_recv_values_norm[i] = 0.0;

        left_last_recv_iters[i] = 0.0;
        right_last_recv_iters[i] = 0.0;

        thres[i] = 0.0;

        for (int j = 0; j < sent_history; j++) {
            sent_slopes_norm[i][j] = 0.0;
        }
    }

    for (int i = 0; i < num_elem_param; i++) {
        left_last_recv_values[i] = 0.0;
        right_last_recv_values[i] = 0.0;

        for (int j = 0; j < recv_history; j++) {
            left_recv_slopes[i][j] = 0.0;
            right_recv_slopes[i][j] = 0.0;
        }
    }

    auto learning_rate = 1e-2;

    torch::optim::SGD optimizer(model->parameters(), learning_rate);

    // File writing
    int file_write = 0;
    char values[30], stats[30], pe_str[3];

    std::ofstream fp;
    std::ofstream fp2;

    sprintf(pe_str, "%d", rank);

    strcpy(values, "values");
    strcat(values, pe_str);
    strcat(values, ".txt");

    strcpy(stats, "stats");
    strcat(stats, pe_str);
    strcat(stats, ".txt");

    if (file_write == 1) {
        fp.open(values);
        fp2.open(stats);
    }
    // end file writing

    // Number of epochs
    auto num_epochs = 250;

    // Number of epochs where event condition not verified (due to starting
    // oscillations)
    auto initial_comm_epochs = 30;

    int num_events = 0;

    // start timer
    tstart = MPI_Wtime();

    for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
        int num_correct = 0;

        for (auto &batch : *data_loader) {
            auto ip = batch.data;
            auto op = batch.target.squeeze();

            // convert to required formats
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);

            // Reset gradients
            model->zero_grad();

            // Execute forward pass
            auto prediction = model->forward(ip);

            // Compute cross entropy loss
            // Direct function not yet supported in libtorch

            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            // Print loss
            if (epoch % 1 == 0 && file_write == 1) {
                // fp << "Output at epoch " << epoch << " = " <<
                // loss.item<float>() << std::endl;
                fp << epoch << ", " << loss.item<float>() << std::endl;
            }

            // Backpropagation
            loss.backward();

            int disp = 0;  // running displacement for RMA window
            for (auto i = 0; i < sz; i++) {
                // getting dimensions of tensor
                int dim0, dim1;
                dim0 = param[i].value().size(0);
                if (param[i].value().dim() > 1) {
                    dim1 = param[i].value().size(1);
                } else {
                    dim1 = 1;
                }

                // flattening the tensor and copying it to a 1-D vector
                auto flat = torch::flatten(param[i].value());

                auto temp = (float *)calloc(flat.numel(),
                                            flat.numel() * param_elem_size);
                for (int j = 0; j < flat.numel(); j++) {
                    *(temp + j) = flat[j].item<float>();
                }

                // consider norm of current parameter
                auto curr_norm = torch::norm(flat).item<float>();
                auto value_diff =
                    std::fabs(curr_norm - last_sent_values_norm[i]);
                auto iter_diff = epoch - last_sent_iters[i];

                thres[i] = thres[i] * std::pow(parameter, iter_diff);

                // SENDING OPERATIONS
                // event - based on norm of current parameter
                if (value_diff >= thres[i] ||
                    epoch < initial_comm_epochs) {
                    num_events += 2;  // for both left and right neighbors


                    // PUSH ENTIRE MSG
                    // send to left
                    MPI_Win_lock(MPI_LOCK_SHARED, left, 0,
                                 win);
                    MPI_Put(temp, flat.numel(), MPI_FLOAT, left,
                            (num_elem_param + disp), flat.numel(), MPI_FLOAT,
                            win);
                    // MPI_Win_flush(left, win);
                    MPI_Win_unlock(left, win);

                    // send to right
                    MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
                    MPI_Put(temp, flat.numel(), MPI_FLOAT, right, disp,
                            flat.numel(), MPI_FLOAT, win);
                    // MPI_Win_flush(right, win);
                    MPI_Win_unlock(right, win);

                    /*
                    //PUSH ONE BY ONE
                    for(int j = 0; j < flat.numel(); j++)
                    {
                       //send to left
                       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, left, 0, win);
                       MPI_Put((temp + j), 1, MPI_FLOAT, left, (num_elem_param +
                    disp + j), 1, MPI_FLOAT, win); MPI_Win_flush(left, win);
                       MPI_Win_unlock(left, win);

                       //send to right
                       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, right, 0, win);
                       MPI_Put((temp + j), 1, MPI_FLOAT, right, (disp + j), 1,
                    MPI_FLOAT, win); MPI_Win_flush(right, win);
                       MPI_Win_unlock(right, win);
                    }
                    */

                    // Shifting previous slope values
                    auto slope_avg = 0.0;
                    int j = 0;
                    for (j = 0; j < sent_history - 1; j++) {
                        sent_slopes_norm[i][j] = sent_slopes_norm[i][j + 1];
                        slope_avg += sent_slopes_norm[i][j];
                    }

                    // Calculating new slope value
                    sent_slopes_norm[i][j] = value_diff / iter_diff;
                    slope_avg += sent_slopes_norm[i][j];
                    slope_avg = slope_avg / sent_history;

                    // Calculating new threshold
                    thres[i] = slope_avg;

                    // update last communicated parameters
                    last_sent_values_norm[i] = curr_norm;
                    last_sent_iters[i] = epoch;
                }

                // RECEIVING OPERATIONS
                // unpack 1-D vector from corresponding displacement and form
                // tensor

                // Left neighbor
                auto left_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                auto left_norm = 0.0;
                for (int j = 0; j < flat.numel(); j++) {
                    *(left_recv + j) = *(win_mem + disp + j);
                    left_norm += std::pow(*(left_recv + j), 2);
                }
                left_norm = std::sqrt(left_norm / flat.numel());

                if (std::fabs(left_norm - left_last_recv_values_norm[i]) > 0) {
                    // new value from left received

                    // shift old values
                    int j, k;
                    for (j = 0; j < flat.numel(); j++) {
                        for (k = 0; k < recv_history; k++) {
                            left_recv_slopes[disp + j][k] =
                                left_recv_slopes[disp + j][k + 1];
                        }
                        left_recv_slopes[disp + j][k] =
                            (left_recv[j] - left_last_recv_values[disp + j]) /
                            (epoch - left_last_recv_iters[i]);
                        left_last_recv_values[disp + j] = left_recv[j];
                    }

                    left_last_recv_values_norm[i] = left_norm;
                    left_last_recv_iters[i] = epoch;
                } else {
                    for (int j = 0; j < flat.numel(); j++) {
                        auto slope_avg = 0.0;
                        for (int k = 0; k < recv_history; k++) {
                            slope_avg += left_recv_slopes[disp + j][k];
                        }
                        slope_avg = slope_avg / recv_history;

                        left_recv[j] =
                            left_last_recv_values[disp + j] +
                            slope_avg * (epoch - left_last_recv_iters[i]);
                    }
                }

                // forming left tensor - either new message or extrapolated
                torch::Tensor left_tensor =
                    torch::from_blob(left_recv, {dim0, dim1}, torch::kFloat)
                        .clone();

                // RIGHT NEIGHBOR
                auto right_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                auto right_norm = 0.0;
                for (int j = 0; j < flat.numel(); j++) {
                    *(right_recv + j) = *(win_mem + num_elem_param + disp + j);
                    right_norm += std::pow(*(right_recv + j), 2);
                }
                right_norm = std::sqrt(right_norm / flat.numel());

                if (std::fabs(right_norm - right_last_recv_values_norm[i]) >
                    0) {
                    // new value from right received

                    // shift old values
                    int j, k;
                    for (j = 0; j < flat.numel(); j++) {
                        for (k = 0; k < recv_history; k++) {
                            right_recv_slopes[disp + j][k] =
                                right_recv_slopes[disp + j][k + 1];
                        }
                        right_recv_slopes[disp + j][k] =
                            (right_recv[j] - right_last_recv_values[disp + j]) /
                            (epoch - right_last_recv_iters[i]);
                        right_last_recv_values[disp + j] = right_recv[j];
                    }

                    right_last_recv_values_norm[i] = right_norm;
                    right_last_recv_iters[i] = epoch;
                } else {
                    for (int j = 0; j < flat.numel(); j++) {
                        auto slope_avg = 0.0;
                        for (int k = 0; k < recv_history; k++) {
                            slope_avg += right_recv_slopes[disp + j][k];
                        }
                        slope_avg = slope_avg / recv_history;

                        right_recv[j] =
                            right_last_recv_values[disp + j] +
                            slope_avg * (epoch - right_last_recv_iters[i]);
                    }
                }

                // forming right tensor - new message or extrapolated
                torch::Tensor right_tensor =
                    torch::from_blob(right_recv, {dim0, dim1}, torch::kFloat)
                        .clone();

                left_tensor.squeeze_();
                right_tensor.squeeze_();

                // averaging with neighbors
                param[i].value().data().add_(left_tensor.data());
                param[i].value().data().add_(right_tensor.data());
                param[i].value().data().div_(3);

                // updating displacement
                disp = disp + flat.numel();

                // freeing temp arrays
                free(temp);
                free(left_recv);
                free(right_recv);
            }

            // Update parameters
            optimizer.step();

            // Accuracy
            auto guess = prediction.argmax(1);
            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();
        }  // end batch loader

        auto accuracy = 100.0 * num_correct / num_train_samples_per_pe;

        if (file_write == 1) fp << epoch << ", " << accuracy << std::endl;

        // Printing parameters to file
        auto param0 = torch::norm(param[0].value()).item<float>();
        auto param1 = torch::norm(param[1].value()).item<float>();
        auto param2 = torch::norm(param[2].value()).item<float>();
        auto param3 = torch::norm(param[3].value()).item<float>();


        if (file_write == 1)
            fp2 << epoch << ", " << param0 << ", " << param1 << ", " << param2
                << ", " << param3 << std::endl;


    }  // end epochs

    // End timer
    tend = MPI_Wtime();
    if (rank == 0)
        std::cout << "Training time - " << (tend - tstart) << std::endl;

    // Print event stats
    // fp2 << "No of events - " << num_events << std::endl;
    std::cout << "No of events in rank " << rank << " - " << num_events
              << std::endl;

    if (file_write == 1) {
        fp.close();
        fp2.close();
    }

    // Averaging learnt model - relevant only for rank 0
    for (int i = 0; i < sz; i++) {
        MPI_Allreduce(MPI_IN_PLACE, param[i].value().data_ptr(),
                      param[i].value().numel(),
                      mpiDatatype.at(param[i].value().scalar_type()), MPI_SUM,
                      MPI_COMM_WORLD);
        if (rank == 0) {
            param[i].value().data() = param[i].value().data() / numranks;
        }
    }

    // Accumulating number of events in all PEs
    MPI_Allreduce(MPI_IN_PLACE, &num_events, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "Total number of events - " << num_events << std::endl;

    // Testing only in rank 0
    if (rank == 0) {
        auto test_dataset =
            torch::data::datasets::MNIST(
                filename, torch::data::datasets::MNIST::Mode::kTest)
                .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                .map(torch::data::transforms::Stack<>());

        auto num_test_samples = test_dataset.size().value();
        auto test_loader = torch::data::make_data_loader(
            std::move(test_dataset), num_test_samples);

        model->eval();

        int num_correct = 0;

        for (auto &batch : *test_loader) {
            auto ip = batch.data;
            auto op = batch.target.squeeze();

            // convert to required format
            ip = ip.to(torch::kF32);
            op = op.to(torch::kLong);

            auto prediction = model->forward(ip);

            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            std::cout << "Test loss - " << loss.item<float>() << " "
                      << std::endl;

            auto guess = prediction.argmax(1);

            // std::cout << "Prediction: " << std::endl << prediction <<
            // std::endl;

            /*
            std::cout << "Output  Guess" << std::endl;
            for(auto i = 0; i < num_test_samples; i++)
            {
                std::cout << op[i].item<int64_t>() << "  " <<
            guess[i].item<int64_t>() << std::endl;
            }
            */

            num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

        }  // end test loader

        std::cout << "Num correct - " << num_correct << std::endl;
        std::cout << "Test Accuracy - "
                  << 100.0 * num_correct / num_test_samples << std::endl;
    }  // end rank 0

    MPI_Finalize();
}

float findThreshold(int epochs, float constant, float gamma)
{
    float thres = 0.0;
    // float gamma = 0.9;
    // float constant = 1e-2;

    thres = constant * pow(gamma, epochs);
    return thres;
}

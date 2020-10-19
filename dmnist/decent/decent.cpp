#include <torch/torch.h>
#include <iostream>
#include "mpi.h"

#define LTAG 2
#define RTAG 10

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

    int file_write = (int)std::atoi(argv[1]);

    // MPI variables
    int rank, numranks;
    int left, right;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    MPI_Request req1, req2;

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
    std::string filename =
        "/afs/crc.nd.edu/user/s/sghosh2/Public/ML/mnist/data";
    auto dataset =
        torch::data::datasets::MNIST(filename)
            .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
            .map(torch::data::transforms::Stack<>());

    // Distributed Random Sampler - use sequential for gradient descent
    auto data_sampler = torch::data::samplers::DistributedSequentialSampler(
        dataset.size().value(), numranks, rank, false);
    auto num_train_samples_per_pe = dataset.size().value() / numranks;

    // Generate dataloader
    int64_t batch_size = num_train_samples_per_pe;
    auto data_loader = torch::data::make_data_loader(std::move(dataset),
                                                     data_sampler, batch_size);

    // setting manual seed - CHECK WHETHER THIS MESSES UP RANDOMNESS IN SGD
    // LATER
    torch::manual_seed(0);

    auto model = std::make_shared<Model>();

    auto sz = model->named_parameters().size();
    auto param = model->named_parameters();
    
    // counting total number of elements in the model
    int num_elem_param = 0;
    for (int i = 0; i < sz; i++) {
        num_elem_param += param[i].value().numel();
    }
    if (rank == 0) {
        std::cout << "Number of parameters - " << sz << std::endl;
        std::cout << "Number of elements - " << num_elem_param << std::endl;
    }
 
    auto param_elem_size = param[0].value().element_size();

    // arrays for storing left and right params
    float left_param[num_elem_param];
    float right_param[num_elem_param];

    // initializing left and right params
    for (int i = 0; i < num_elem_param; i++) {
        left_param[i] = 0.0;
        right_param[i] = 0.0;
    }

    auto learning_rate = 1e-2;

    torch::optim::SGD optimizer(model->parameters(), learning_rate);

    // File writing
    char name[30], pe_str[3];
    std::ofstream fp;
    sprintf(pe_str, "%d", rank);
    strcpy(name, "values");
    strcat(name, pe_str);
    strcat(name, ".txt");

    if (file_write == 1) {
        fp.open(name);
    }
    // end file writing

    // Number of epochs
    auto num_epochs = 50; //250;

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
            auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

            // Print loss
            if (epoch % 1 == 0 && file_write == 1) {
                fp << epoch << ", " << loss.item<float>() << std::endl;
            }

            // Backpropagation
            loss.backward();

            int disp = 0;  // displacement of left and right params
            for (auto i = 0; i < sz; i++) {
                // getting dimensions of tensor

                int num_dim = param[i].value().dim();
                std::vector<int64_t> dim_array;
                for (int j = 0; j < num_dim; j++) {
                    dim_array.push_back(param[i].value().size(j));
                }

                // flattening the tensor and copying it to a 1-D vector
                auto flat = torch::flatten(param[i].value());

                auto temp = (float *)calloc(flat.numel(),
                                            flat.numel() * param_elem_size);
                for (int j = 0; j < flat.numel(); j++) {
                    *(temp + j) = flat[j].item<float>();
                }

                // send parameters to left
                MPI_Issend(temp, flat.numel(), MPI_FLOAT, left, RTAG,
                           MPI_COMM_WORLD, &req1);

                // send parameters to right
                MPI_Issend(temp, flat.numel(), MPI_FLOAT, right, LTAG,
                           MPI_COMM_WORLD, &req2);

                // receive from left
                MPI_Recv((left_param + disp), flat.numel(), MPI_FLOAT, left,
                         LTAG, MPI_COMM_WORLD, &status);

                // receive from right
                MPI_Recv((right_param + disp), flat.numel(), MPI_FLOAT, right,
                         RTAG, MPI_COMM_WORLD, &status);

                MPI_Wait(&req1, &status);
                MPI_Wait(&req2, &status);

                // unpack 1-D vector form corresponding displacement and form
                // tensor
                auto left_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                // fp << "left - " << std::endl;
                for (int j = 0; j < flat.numel(); j++) {
                    *(left_recv + j) = *(left_param + disp + j);
                }
                torch::Tensor left_tensor =
                    torch::from_blob(left_recv, dim_array, torch::kFloat)
                        .clone();

                auto right_recv = (float *)calloc(
                    flat.numel(), flat.numel() * param_elem_size);
                for (int j = 0; j < flat.numel(); j++) {
                    *(right_recv + j) = *(right_param + disp + j);
                }
                torch::Tensor right_tensor =
                    torch::from_blob(right_recv, dim_array, torch::kFloat)
                        .clone();

                // average gradients
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

        std::cout << epoch << ", " << accuracy << std::endl;

        /*
        // Printing parameters to file
        auto param0 = torch::norm(param[0].value()).item<float>();
        auto param1 = torch::norm(param[1].value()).item<float>();
        auto param2 = torch::norm(param[2].value()).item<float>();
        auto param3 = torch::norm(param[3].value()).item<float>();

        if (file_write == 1)
            fp << epoch << ", " << param0 << ", " << param1 << ", " << param2
               << ", " << param3 << std::endl;
        */

    }  // end epochs

    // end timer
    tend = MPI_Wtime();
    if (rank == 0)
        std::cout << "Training time - " << (tend - tstart) << std::endl;

    if (file_write == 1) fp.close();

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

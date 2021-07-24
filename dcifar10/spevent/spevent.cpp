#include "../common/custom.hpp"
#include "../common/nnet.hpp"
#include "../common/resnet.hpp"
#include "../common/transform.hpp"
#include "mpi.h"
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <stdint.h>
#include <string>
#include <torch/torch.h>
#include <vector>
#include <cmath>

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;

std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

struct Options {
  size_t train_samples = 50000;
  size_t train_batch_size = 256;

  size_t test_samples = 10000;
  size_t test_batch_size = 100;

  size_t iterations = 20;
  size_t log_interval = 20;

  torch::DeviceType device = (torch::kCPU);
};

static Options options;

int main(int argc, char *argv[]) {

  int file_write = (int)std::atoi(argv[1]);
  int thres_type =
      (int)std::atoi(argv[2]); // 0 for non-adaptive, 1 for adaptive

  float horizon, constant;

  if (thres_type == 1) {
    horizon = (float)std::atof(argv[3]); // adaptive threshold
  } else {
    constant = (float)std::atof(argv[3]); // non-adaptive constant
                                          // threshold
  }

  int topk_percent = (int)std::atoi(argv[4]); // top-k percentage

  // history at sender
  auto sent_history = 2;

  // MPI variables
  int rank, numranks;
  int left, right;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Win win;

  if (numranks > 1) { // excluding serial case
    // ring arrangement
    if (rank == 0)
      left = numranks - 1;
    else
      left = rank - 1;

    if (rank == numranks - 1)
      right = 0;
    else
      right = rank + 1;
  }

  // end MPI variables

  // Timer variables
  auto tstart = 0.0;
  auto tend = 0.0;

  auto train_samples_per_pe = options.train_samples / numranks;

  auto train_batch_per_pe = options.train_batch_size / numranks;

  auto data = readInfo();
  auto train_set = CustomDataset(data.first)
                       .map(ConstantPad(4))
                       .map(RandomHorizontalFlip())
                       .map(RandomCrop({32, 32}))
                       .map(torch::data::transforms::Stack<>());

  auto train_size = train_set.size().value();

  auto train_sampler = torch::data::samplers::DistributedRandomSampler(
      train_size, numranks, rank, false);
  auto train_loader = torch::data::make_data_loader(
      std::move(train_set), train_sampler, train_batch_per_pe);

  auto test_set =
      CustomDataset(data.second).map(torch::data::transforms::Stack<>());
  auto test_size = test_set.size().value();
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_set), options.test_batch_size);

  // sets manual seed
  torch::manual_seed(0);

  // auto model = std::make_shared<Model>();

  std::array<int64_t, 4> layers{2, 2, 2, 2};
  ResNet<BasicBlock> model(layers, 10);

  auto sz = model.named_parameters().size();
  auto param = model.named_parameters();

  // Dummy model to keep previous values
  ResNet<BasicBlock> prev_model(layers, 10);
  auto prev_param = prev_model.named_parameters();

  // Dummy left and right models as well
  ResNet<BasicBlock> left_model(layers, 10);
  ResNet<BasicBlock> right_model(layers, 10);
  auto left_param = left_model.named_parameters();
  auto right_param = right_model.named_parameters();  

  // counting total number of elements in the model
  int num_elem_param = 0;
  for (auto i = 0; i < sz; i++) {
    num_elem_param += param[i].value().numel();
  }

  // counting number of top-k elements
  int num_topk_param = 0; // total topk
  int topk_per_param[sz]; // topk per param
  for (auto i = 0; i < sz; i++) {
    topk_per_param[i] = (int)ceil((topk_percent/100.0) * param[i].value().numel());
    num_topk_param += topk_per_param[i];
  }

  if (rank == 0) {
    std::cout << "Number of parameters - " << sz << std::endl;
    std::cout << "Number of elements - " << num_elem_param << std::endl;
    std::cout << "Number of topk elements - " << num_topk_param << std::endl;
  }

  auto param_elem_size = param[0].value().element_size();

  // create memory window for parameters from left and right
  auto win_size = 4 * num_topk_param; // value and index for each neighbor
  float *win_mem;
  MPI_Alloc_mem(win_size * param_elem_size, MPI_INFO_NULL, &win_mem);
  MPI_Win_create(win_mem, win_size * param_elem_size, param_elem_size,
                 MPI_INFO_NULL, MPI_COMM_WORLD, &win);

  // initializing RMA window
  for (int i = 0; i < win_size; i++) {
    *(win_mem + i) = 0.0;
  }

  // Threshold
  auto thres = (float *)calloc(sz, param_elem_size);

  // variables at the sender
  auto last_sent_values_norm = (float *)calloc(sz, param_elem_size);
  auto last_sent_iters = (float *)calloc(sz, param_elem_size);
  auto sent_slopes_norm = (float *)calloc(sz * sent_history, param_elem_size);

  // variables at the receiver
  auto left_last_recv_values = (float *)calloc(num_elem_param, param_elem_size);
  auto left_last_recv_values_norm = (float *)calloc(sz, param_elem_size);
  auto left_last_recv_iters = (float *)calloc(sz, param_elem_size);

  auto right_last_recv_values =
      (float *)calloc(num_elem_param, param_elem_size);
  auto right_last_recv_values_norm = (float *)calloc(sz, param_elem_size);
  auto right_last_recv_iters = (float *)calloc(sz, param_elem_size);

  auto left_recv_norm = (float *)calloc(sz, param_elem_size);
  auto right_recv_norm = (float *)calloc(sz, param_elem_size);

  // initializing values
  for (int i = 0; i < sz; i++) {
    *(last_sent_values_norm + i) = 0.0;
    *(last_sent_iters + i) = 0.0;

    *(left_last_recv_values_norm + i) = 0.0;
    *(right_last_recv_values_norm + i) = 0.0;

    *(left_last_recv_iters + i) = 0.0;
    *(right_last_recv_iters + i) = 0.0;

    *(left_recv_norm + i) = 0.0;
    *(right_recv_norm + i) = 0.0;

    *(thres + i) = 0.0;

    for (int j = 0; j < sent_history; j++) {
      *(sent_slopes_norm + i * sent_history + j) = 0.0;
    }
  }

  for (int i = 0; i < num_elem_param; i++) {
    *(left_last_recv_values + i) = 0.0;
    *(right_last_recv_values + i) = 0.0;
  }

  auto learning_rate = 1e-2;

  torch::optim::SGD optimizer(
      model.parameters(),
      torch::optim::SGDOptions(learning_rate).momentum(0.9));

  // File writing
  char train_name[30], send_name[30], recv_name[30], pe_str[3];

  std::ofstream fpt; // file for printing training stats
  std::ofstream fps; // file for sending log
  std::ofstream fpr; // file for receiving log

  sprintf(pe_str, "%d", rank);

  strcpy(train_name, "train");
  strcat(train_name, pe_str);
  strcat(train_name, ".txt");

  strcpy(send_name, "send");
  strcat(send_name, pe_str);
  strcat(send_name, ".txt");

  strcpy(recv_name, "recv");
  strcat(recv_name, pe_str);
  strcat(recv_name, ".txt");

  if (file_write == 1) {
    fpt.open(train_name);
    fps.open(send_name);
    fpr.open(recv_name);
  }
  // end file writing

  // Number of epochs
  auto num_epochs = options.iterations;

  // Previous epochs + pass number through current epoch
  auto pass_num = 0;

  // Number of epochs where event condition not verified (due to starting
  // oscillations)
  auto initial_comm_passes = 30;

  int num_events = 0;

  // start timer
  tstart = MPI_Wtime();

  for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
    int num_correct = 0;

    for (auto &batch : *train_loader) {
      pass_num++;

      auto ip = batch.data;
      auto op = batch.target.squeeze();

      // convert to required formats
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      // Reset gradients
      model.zero_grad();
     
      prev_model.zero_grad();
      left_model.zero_grad();
      right_model.zero_grad();

      // Execute forward pass
      auto prediction = model.forward(ip);

      // Compute cross entropy loss

      //auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);
      auto loss = torch::nn::functional::cross_entropy(prediction, op);

      // Print loss
      if (file_write == 1) {
        fpt << pass_num << ", " << loss.item<float>() << std::endl;
      }

      // std::cout << "Pass Num in rank " << rank << " = " << pass_num << std::endl;

      // Backpropagation
      loss.backward();

      int disp = 0; // running displacement for RMA window

      // parameter loop
      if (numranks > 1) {
        for (auto i = 0; i < sz; i++) {
          // getting dimensions of tensor

          int num_dim = param[i].value().dim();
          std::vector<int64_t> dim_array;
          for (int j = 0; j < num_dim; j++) {
            dim_array.push_back(param[i].value().size(j));
          }

          auto flat = torch::flatten(param[i].value()).contiguous();

          // consider norm of current parameter
          auto curr_norm = torch::norm(flat).item<float>();
          auto value_diff = std::fabs(curr_norm - *(last_sent_values_norm + i));
          auto iter_diff = pass_num - *(last_sent_iters + i);

          if (thres_type == 1) {
            *(thres + i) = *(thres + i) * horizon;
          } else {
            *(thres + i) = constant;
          }

          // Printing value of norm of current parameter
          if (file_write == 1) {
            fps << curr_norm << ",  " << *(thres + i) << ",  ";
          }

          // SENDING OPERATIONS
          // event - based on norm of current parameter
          if (value_diff >= *(thres + i) || pass_num < initial_comm_passes) {
            num_events += 2; // for both left and right neighbors

            // GENERATE TOP-K MESSAGE
            // flattening corresponding prev model tensor
            auto prev_flat = torch::flatten(prev_param[i].value()).contiguous();

            auto diff = torch::abs(torch::sub(flat, prev_flat, 1)); // absolute difference

            // Top-k operation
            auto topk_tuple = torch::topk(diff, topk_per_param[i], 0, true, true);
            auto indices_topk = std::get<1>(topk_tuple);
            indices_topk = indices_topk.to(torch::kFloat);

            float *elem_temp = (float *)calloc(topk_per_param[i], param_elem_size);
            float *indices_temp = (float *)calloc(topk_per_param[i], param_elem_size);

            // Generate the array with indices
            std::memcpy(indices_temp, indices_topk.data_ptr(), topk_per_param[i] * param_elem_size);

            // Get the array with tensor elements at those indices
            for (int k = 0; k < topk_per_param[i]; k++) {
               int ind = (int)*(indices_temp + k);
               *(elem_temp + k) = flat.data()[ind].item<float>();
            }  

            // PUSH ELEMENTS AND INDICES
            // send to left
            MPI_Win_lock(MPI_LOCK_SHARED, left, 0, win);
            MPI_Put(elem_temp, topk_per_param[i], MPI_FLOAT, left, (2 * num_topk_param + disp),
                    topk_per_param[i], MPI_FLOAT, win);
            MPI_Put(indices_temp, topk_per_param[i], MPI_FLOAT, left, 
                   (2 * num_topk_param + disp + topk_per_param[i]), topk_per_param[i], MPI_FLOAT, win);
            // MPI_Win_flush(left, win);
            MPI_Win_unlock(left, win);

            // send to right
            MPI_Win_lock(MPI_LOCK_SHARED, right, 0, win);
            MPI_Put(elem_temp, topk_per_param[i], MPI_FLOAT, right, disp, topk_per_param[i], MPI_FLOAT, win);
            MPI_Put(indices_temp, topk_per_param[i], MPI_FLOAT, right, (disp + topk_per_param[i]),
                    topk_per_param[i], MPI_FLOAT, win);
            // MPI_Win_flush(right, win);
            MPI_Win_unlock(right, win);

            // Shifting previous slope values
            auto slope_avg = 0.0;
            int j = 0;
            for (j = 0; j < sent_history - 1; j++) {
              *(sent_slopes_norm + i * sent_history + j) =
                  *(sent_slopes_norm + i * sent_history + j + 1);
              slope_avg += *(sent_slopes_norm + i * sent_history + j);
            }

            // Calculating new slope value
            *(sent_slopes_norm + i * sent_history + j) = value_diff / iter_diff;
            slope_avg += *(sent_slopes_norm + i * sent_history + j);
            slope_avg = slope_avg / sent_history;

            // Calculating new threshold if adaptive
            if (thres_type == 1) {
              *(thres + i) = slope_avg;
            }

            // update last communicated parameters
            *(last_sent_values_norm + i) = curr_norm;
            *(last_sent_iters + i) = pass_num;

            // update prev model - copy only the topk values being sent
            for (int k = 0; k < topk_per_param[i]; k++) {
               int ind = (int)*(indices_temp + k);
               prev_flat.data()[ind] = *(elem_temp + k); // change the top-k elements in prev_flat
            }    
            // reshape prev flat
            prev_flat = torch::reshape(prev_flat, dim_array);
            prev_param[i].value().data().copy_(prev_flat.data());

            // record that an event was triggered
            if (file_write == 1) {
              fps << "1,  ";
            }

            free(elem_temp);
            free(indices_temp);
          } else {
            if (file_write == 1) {
              fps << "0,  ";
            }
          } // end send

          // RECEIVING OPERATIONS
          // unpack 1-D vector from corresponding displacement and form
          // tensor

          // Left neighbor
          float *left_recv = (float *)calloc(2 * topk_per_param[i], param_elem_size);
          float left_temp = 0.0;
          std::memcpy(left_recv, (win_mem + disp), 2 * topk_per_param[i] * param_elem_size);
  
          // Temp flat tensor
          auto left_tensor = torch::flatten(left_param[i].value()).contiguous();
          // Copy the values at topk indices to this tensor
          for (int k = 0; k < topk_per_param[i]; k++) {
               int ind = (int)*(left_recv + topk_per_param[i] + k);
               left_tensor.data()[ind] = *(left_recv + k); // copy to the temp tensor
          }        

          // Reshaping tensor
          left_tensor = torch::reshape(left_tensor, dim_array);
          // Copy it back to left_model
          left_param[i].value().data().copy_(left_tensor.data());

          left_temp = torch::norm(left_tensor).item<float>();
          *(left_recv_norm + i) = left_temp;

          /*
          for (int j = 0; j < flat.numel(); j++) {
            *(left_recv + j) = *(win_mem + disp + j);
            left_temp += std::pow(*(left_recv + j), 2);
          }
          left_temp = std::sqrt(left_temp / flat.numel());
          */

          auto left_recv_diff = std::fabs(*(left_recv_norm + i) -
                                          *(left_last_recv_values_norm + i));

          if (left_recv_diff > 0) {
            // new value from left received

            *(left_last_recv_values_norm + i) = *(left_recv_norm + i);
            *(left_last_recv_iters + i) = pass_num;

            // Record that new value is received
            if (file_write == 1) {
              fpr << "1,  ";
            }
          } else {
            if (file_write == 1) {
              fpr << "0,  ";
            }
          }

          // Writing value received
          if (file_write == 1) {
            fpr << left_temp << ",  "; // << left_recv_diff << ",  ";
          }

          // Right Neighbor
          float *right_recv = (float *)calloc(2 * topk_per_param[i], param_elem_size);
          float right_temp = 0.0;
          std::memcpy(right_recv, (win_mem + 2 * num_topk_param + disp),
                      2 * topk_per_param[i] * param_elem_size);

          // Temp flat tensor
          auto right_tensor = torch::flatten(right_param[i].value()).contiguous();
          // Copy the values at topk indices to this tensor
          for (int k = 0; k < topk_per_param[i]; k++) {
             int ind = (int)*(right_recv + topk_per_param[i] + k);
             right_tensor.data()[ind] = *(right_recv + k);
          }

          // Reshaping tensor
          right_tensor = torch::reshape(right_tensor, dim_array);
          // Copy it back to right_model
          right_param[i].value().data().copy_(right_tensor.data());

          right_temp = torch::norm(right_tensor).item<float>();
          *(right_recv_norm + i) = right_temp;

          /*
          for (int j = 0; j < flat.numel(); j++) {
            *(right_recv + j) = *(win_mem + num_elem_param + disp + j);
            right_temp += std::pow(*(right_recv + j), 2);
          }
          right_temp = std::sqrt(right_temp / flat.numel());
          */

          auto right_recv_diff = std::fabs(*(right_recv_norm + i) -
                                           *(right_last_recv_values_norm + i));

          if (right_recv_diff > 0) {
            // new value from right received

            *(right_last_recv_values_norm + i) = *(right_recv_norm + i);
            *(right_last_recv_iters + i) = pass_num;

            // record that new value is received
            if (file_write == 1) {
              fpr << "1,  ";
            }
          } else {
            if (file_write == 1) {
              fpr << "0,  ";
            }
          }

          // writing value received
          if (file_write == 1) {
            fpr << right_temp << ",  "; // << right_recv_diff << ",  ";
          }

          // averaging with neighbors
          param[i].value().data().add_(left_tensor.data());
          param[i].value().data().add_(right_tensor.data());
          param[i].value().data().div_(3);

          // updating displacement
          disp = disp + 2 * topk_per_param[i];

          // freeing temp arrays
          free(left_recv);
          free(right_recv);
        } // end parameter loop
      }   // end numranks > 1

      if (file_write == 1) {
        fps << std::endl;
        fpr << std::endl;
      }

      // Update parameters
      optimizer.step();

      // Accuracy
      auto guess = prediction.argmax(1);
      num_correct += torch::sum(guess.eq_(op)).item<int64_t>();

    } // end batch loader

    auto accuracy = 100.0 * num_correct / train_samples_per_pe;

    std::cout << "Accuracy in epoch " << epoch << " - " << accuracy
              << std::endl;

  } // end epochs

  // End timer
  tend = MPI_Wtime();
  if (rank == 0)
    std::cout << "Training time - " << (tend - tstart) << std::endl;

  // Print event stats
  std::cout << "No of events in rank " << rank << " - " << num_events
            << std::endl;

  if (file_write == 1) {
    fpt.close();
    fps.close();
    fpr.close();
  }

  if (numranks > 1) {
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
  } // end numranks > 1

  if (rank == 0) {
    std::cout << "Total number of events - " << num_events << std::endl;
  }

  // TESTING
  model.eval();

  // Testing only in rank 0
  if (rank == 0) {

    int num_correct = 0;

    for (auto &batch : *test_loader) {
      auto ip = batch.data;
      auto op = batch.target.squeeze();

      // convert to required format
      ip = ip.to(torch::kF32);
      op = op.to(torch::kLong);

      auto prediction = model.forward(ip);

      auto loss = torch::nll_loss(torch::log_softmax(prediction, 1), op);

      // std::cout << "Test loss - " << loss.item<float>() << " " << std::endl;

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

    } // end test loader

    std::cout << "Num correct - " << num_correct << std::endl;
    std::cout << "Test Accuracy - "
              << 100.0 * num_correct / options.test_samples << std::endl;
  } // end rank 0

  MPI_Finalize();
}

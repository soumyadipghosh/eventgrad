#include <torch/torch.h>
#include <iostream>
#include "custom_dataset.hpp"
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

int main(int argc, char *argv[])
{
  auto D_in = 4; //dimension of input layer
  auto H = 4; //dimension of hidden layer
  auto D_out = 3; //dimension of output layer

  //MPI variables
  int rank, numranks;
  int left, right;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;
  MPI_Request req1, req2;

  //ring arrangement
  if(rank == 0)
    left = numranks - 1;
  else
    left = rank - 1;

  if(rank == numranks - 1)
    right = 0;
  else
    right = rank + 1;

  //end MPI variables

  //Read dataset
  std::string filename = "../../cpp-examples/datasets/iris_train.csv";
  auto dataset = CustomDataset(filename).map(torch::data::transforms::Stack<>());
  
  //Distributed Random Sampler
  auto data_sampler = torch::data::samplers::DistributedRandomSampler(dataset.size().value(),
                                             numranks, rank, false);

  //Generate dataloader
  int64_t batch_size = 30; //batch size
  auto data_loader = torch::data::make_data_loader
                     (std::move(dataset), data_sampler, batch_size);

  auto model = torch::nn::Sequential(
      torch::nn::Linear(D_in, H),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(H, D_out));

  //keep a copy of the model for left and right for storing gradients
  auto left_model = model;
  auto right_model = model;

  auto learning_rate = 1e-2;

  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  //File writing
  int file_write = 1;
  char name[30], pe_str[3];
  std::ofstream fp;
  sprintf(pe_str, "%d", rank);
  strcpy(name, "values");
  strcat(name, pe_str);
  strcat(name, ".txt");

  if(file_write == 1)
  {
    fp.open(name);
  }
  //end file writing

  for (size_t epoch = 1; epoch <= 100; ++epoch)
  {
      //size_t batch_idx = 0;
      //float mse = 0;       //mse
      //int count = 0;

      for (auto& batch : *data_loader)
      {

          auto ip = batch.data;
          auto op = batch.target.squeeze();

          //convert to required formats
          ip = ip.to(torch::kF32); 
          op = op.to(torch::kLong);

	  //Reset gradients
	  optimizer.zero_grad();

	  //Execute forward pass
	  auto prediction = model->forward(ip);

          //Compute cross entropy loss
          //Direct function not yet supported in libtorch

          //auto loss = torch::mse_loss(prediction, op);
          auto loss = torch::nll_loss(torch::log_softmax(prediction, 1) , op);

          //Print loss
          if (epoch % 1 == 0)
	      fp << "Output at epoch " << epoch << " = " << loss.item<float>() << std::endl;

	  //Backpropagation
	  loss.backward();

          auto sz = model->named_parameters().size();
          auto param = model->named_parameters();
          auto left_param = left_model->named_parameters();
          auto right_param = right_model->named_parameters();

          for(auto i = 0; i < sz; i++)
          {
              //send gradients to left
              MPI_Isend(param[i].value().grad().data_ptr(), param[i].value().grad().numel(),
                        mpiDatatype.at(param[i].value().grad().scalar_type()),
                        left, RTAG, MPI_COMM_WORLD, &req1);

              //send gradients to right
              MPI_Isend(param[i].value().grad().data_ptr(), param[i].value().grad().numel(),
                        mpiDatatype.at(param[i].value().grad().scalar_type()),
                        right, LTAG, MPI_COMM_WORLD, &req2);

              MPI_Wait(&req1, &status);
              MPI_Wait(&req2, &status);
              
              MPI_Recv(left_param[i].value().grad().data_ptr(), left_param[i].value().grad().numel(),
                       mpiDatatype.at(left_param[i].value().grad().scalar_type()),
                       left, LTAG, MPI_COMM_WORLD, &status);

              MPI_Recv(right_param[i].value().grad().data_ptr(), right_param[i].value().grad().numel(),
                       mpiDatatype.at(right_param[i].value().grad().scalar_type()),
                       right, RTAG, MPI_COMM_WORLD, &status); 

              //fp << param[i].key() << "-  " << torch::norm(param[i].value()).item<float>() << std::endl;
              fp << torch::norm(param[i].value().grad()).item<float>() << std::endl;
              //fp << param[i].value().grad() << std::endl; //same as grad().data()
              //fp << param[i].value().grad().data() << std::endl;

              //average gradients     
              param[i].value().grad().add_(left_param[i].value().grad());
              param[i].value().grad().add_(right_param[i].value().grad());    
              param[i].value().grad().div_(3);
     
              fp << torch::norm(param[i].value().grad()).item<float>() << std::endl;
          } 
          
	  //Update parameters
	  optimizer.step();
      } 
   }

   if(file_write == 1) fp.close();

   MPI_Finalize();
}

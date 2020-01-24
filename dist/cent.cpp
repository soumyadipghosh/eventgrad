#include <torch/torch.h>
#include <iostream>
#include "custom_dataset.hpp"
#include "mpi.h"

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
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;
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

  for (size_t epoch = 1; epoch <= 20; ++epoch)
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

          //Print model weights
          for (auto &param : model->named_parameters())
          {
              //fp << param.key() << "-  " << torch::norm(param.value()).item<float>() << std::endl;
              fp << torch::norm(param.value().grad()).item<float>() << std::endl;

              MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(),
               param.value().grad().numel(),
                mpiDatatype.at(param.value().grad().scalar_type()),
                MPI_SUM, MPI_COMM_WORLD);

              param.value().grad().data() = param.value().grad().data()/numranks;

              //fp << torch::norm(param.value().grad()).item<float>() << std::endl;
          } 

	  //Update parameters
	  optimizer.step();
      } 
   }

   if(file_write == 1) fp.close();

   MPI_Finalize();
}

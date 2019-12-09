#include <torch/torch.h>
#include <iostream>
#include <fstream>
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

   auto N = 64; //batch size - number of input samples
   auto D_in = 3; //i/p dimension - no of features in each i/p sample
   auto H = 3; //hidden dimension
   auto D_out = 1; //o/p dimension

   auto x = torch::randn({N, D_in});
   auto y = torch::randn({N, D_out});

   auto model = torch::nn::Sequential(
      torch::nn::Linear(D_in, H),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(H, D_out));

   //MPI variables
   int rank, numranks;
   int batch_size = 1;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numranks);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Status status;
   //end MPI variables

   int i, j;
   int file_write = 1;
   char name[30], pe_str[3];
   
   auto learning_rate  = 1e-4;
   auto num_batches = D_in / batch_size;
   torch::optim::SGD optimizer(model->parameters(), learning_rate);
   
   //File writing
   std::ofstream fp;
   sprintf(pe_str, "%d", rank);
   strcpy(name, "values");
   strcat(name, pe_str);
   strcat(name, ".txt");

   if(file_write == 1)
   {
      fp.open(name);
   }

   for(size_t epoch = 1; epoch <= 5; ++epoch)
   {
      //Reset gradients
      optimizer.zero_grad();

      //Execute forward pass
      torch::Tensor prediction = model->forward(x);

      //Compute loss
      auto loss = torch::mse_loss(prediction, y);

      //Backpropagate loss
      loss.backward();

      //Average gradients
      
      for (auto &param : model->named_parameters())
      {
          MPI_Allreduce(MPI_IN_PLACE, param.value().grad().data_ptr(),
               param.value().grad().numel(),
                mpiDatatype.at(param.value().grad().scalar_type()),
                MPI_SUM, MPI_COMM_WORLD);

          param.value().grad().data() = param.value().grad().data()/numranks;

      }
      
      /*
      MPI_Allreduce(MPI_IN_PLACE, model[0].weight.grad,
               model[0].weight.grad.numel(), mpiDatatype.at(tensor.scalar_type()),
                MPI_SUM, MPI_COMM_WORLD);
      */
      //Update parameters
      optimizer.step();

      if(file_write == 1)
      {
	  //Print model weights
	  for (auto &p : model->named_parameters())
	  {
	      fp << p.key() << "-  " << torch::norm(p.value()).item<float>() << std::endl;
	  }

	  //Print loss
	  if (epoch % 1 == 0)
	      fp << "Loss at epoch " << epoch << " = " << loss.item<float>() << std::endl;
      }
      
    }
    if(file_write == 1) fp.close();

    MPI_Finalize();
}

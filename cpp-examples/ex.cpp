#include <torch/torch.h>
#include <iostream>

int main(int argc, char *argv[])
{

  auto N = 1;
  auto D_in = 3;
  auto H = 3;
  auto D_out = 1;

  auto x = torch::randn({N, D_in});
  auto y = torch::randn({N, D_out});

  auto model = torch::nn::Sequential(
      torch::nn::Linear(D_in, H),
      torch::nn::Functional(torch::relu),
      torch::nn::Linear(H, D_out));

  auto learning_rate = 1e-4;

  torch::optim::SGD optimizer(model->parameters(), learning_rate);

  for (size_t epoch = 1; epoch <= 5; ++epoch)
  {
	//Reset gradients
	optimizer.zero_grad();

	//Execute forward pass
	torch::Tensor prediction = model->forward(x);

        //Compute loss
        auto loss = torch::mse_loss(prediction, y);

        //Print loss
        if (epoch % 1 == 0)
		std::cout << "Loss at epoch " << epoch << " = " << loss.item<float>() << std::endl;

        //Print model weights
        for (auto &p : model->named_parameters())
        {
            std::cout << p.key() << "-  " << torch::norm(p.value()).item<float>() << std::endl;
        }

	//Backpropagation
	loss.backward();

	//Update parameters
	optimizer.step();
  }
}



#include <torch/torch.h>

struct Model : torch::nn::Module {
  Model()
      : conv1(torch::nn::Conv2dOptions(3, 6, 5)),
        conv2(torch::nn::Conv2dOptions(6, 16, 5)), fc1(16 * 5 * 5, 120),
        fc2(120, 84), fc3(84, 10) {
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("conv2_drop", conv2_drop);
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
  }
  
  torch::Tensor forward(torch::Tensor x) {
    x = torch::max_pool2d(torch::relu(conv1->forward(x)), {2, 2});
    x = torch::max_pool2d(torch::relu(conv2_drop->forward(conv2->forward(x))),
                          {2, 2});
    x = x.view({-1, 16 * 5 * 5});
    x = torch::relu(fc1->forward(x));
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return torch::log_softmax(x, 1);
  }
  
  torch::nn::Conv2d conv1;
  torch::nn::Conv2d conv2;
  torch::nn::Dropout2d conv2_drop;
  torch::nn::Linear fc1;
  torch::nn::Linear fc2;
  torch::nn::Linear fc3;
};


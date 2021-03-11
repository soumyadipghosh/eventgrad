#include <torch/torch.h>

torch::nn::Conv2d conv_op(int64_t in_channels, int64_t out_channels, int64_t kernel_size, 
                          int64_t stride, int padding) {
    return torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
        .stride(stride)
        .padding(padding)
        .bias(false));
}

struct BasicBlock : torch::nn::Module {

  static const int expansion;

  //int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  torch::nn::Sequential downsampler;

  BasicBlock(int64_t in_channels, int64_t out_channels, int64_t stride=1,
             torch::nn::Sequential downsample=nullptr)
      : conv1(conv_op(in_channels, out_channels, 3, stride, 1)),
        bn1(out_channels),
        conv2(conv_op(out_channels, out_channels, 3, 1, 1)),
        bn2(out_channels),
        downsampler(downsample) {
      register_module("conv1", conv1);
      register_module("bn1", bn1);
      register_module("conv2", conv2);
      register_module("bn2", bn2);

      if (downsampler) {
         register_module("downsampler", downsampler);
      }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);
    out = conv2->forward(out);
    out = bn2->forward(out);

    auto residual = downsampler ? downsampler->forward(x) : x;
    out += residual;
    out = torch::relu(out);

    return out;
  }
};

const int BasicBlock::expansion = 1;

struct BottleNeck : torch::nn::Module {
  
  static const int expansion;

  //int64_t stride;
  torch::nn::Conv2d conv1;
  torch::nn::BatchNorm2d bn1;
  torch::nn::Conv2d conv2;
  torch::nn::BatchNorm2d bn2;
  torch::nn::Conv2d conv3;
  torch::nn::BatchNorm2d bn3;
  torch::nn::Sequential downsampler;

  BottleNeck(int64_t in_channels, int64_t out_channels, int64_t stride=1,
             torch::nn::Sequential downsample=nullptr)
      : conv1(conv_op(in_channels, out_channels, 1, 1, 0)),
        bn1(out_channels),
        conv2(conv_op(out_channels, out_channels, 3, stride, 1)),
        bn2(out_channels),
        conv3(conv_op(out_channels, out_channels * expansion, 1, 1, 0)),
        bn3(out_channels * expansion),
        downsampler(downsample)
        {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    register_module("conv3", conv3);
    register_module("bn3", bn3);
    
    if (downsampler) {
      register_module("downsampler", downsampler);
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    auto out = conv1->forward(x);
    out = bn1->forward(out);
    out = torch::relu(out);
    out = conv2->forward(out);
    out = bn2->forward(out);
    out = torch::relu(out);
    out = conv3->forward(out);
    out = bn3->forward(out);

    auto residual = downsampler ? downsampler->forward(x) : x;
    out += residual;
    out = torch::relu(out);

    return out;
  }
};

const int BottleNeck::expansion = 4;

template <class Block> struct ResNet : torch::nn::Module {
  
  int64_t in_channels = 64;
  torch::nn::Conv2d conv;
  torch::nn::BatchNorm2d bn;
  torch::nn::Sequential layer1;
  torch::nn::Sequential layer2;
  torch::nn::Sequential layer3;
  torch::nn::Sequential layer4;
  torch::nn::Linear fc;

  ResNet(const std::array<int64_t, 4>& layers, int64_t num_classes)
      : conv(conv_op(3, 64, 3, 1, 1)),
        bn(64),
        layer1(make_layer(64, layers[0], 1)),
        layer2(make_layer(128, layers[1], 2)),
        layer3(make_layer(256, layers[2], 2)),
        layer4(make_layer(512, layers[3], 2)),
        fc(512 * Block::expansion, num_classes)
        {
     register_module("conv", conv);
     register_module("bn", bn);
     register_module("layer1", layer1);
     register_module("layer2", layer2);
     register_module("layer3", layer3);
     register_module("layer4", layer4);
     register_module("fc", fc);

  }

  torch::Tensor forward(torch::Tensor x) {
    auto out = conv->forward(x);
    out = bn->forward(out);
    out = torch::relu(out);
    //out = torch::max_pool2d(out, 3, 2, 1);

    out = layer1->forward(out);
    out = layer2->forward(out);
    out = layer3->forward(out);
    out = layer4->forward(out);

    out = torch::avg_pool2d(out, 4);
    out = out.view({out.size(0), -1});
    out = fc->forward(out);

    return out;
  }

private:
  torch::nn::Sequential make_layer(int64_t out_channels, int64_t blocks, int64_t stride) {

    torch::nn::Sequential layers;
    torch::nn::Sequential downsample{nullptr};

    if (stride != 1 || in_channels != out_channels * Block::expansion) {
      downsample = torch::nn::Sequential{
          conv_op(in_channels, out_channels * Block::expansion, 1, stride, 0),
          torch::nn::BatchNorm2d(out_channels * Block::expansion)
      };
    }
    
    layers->push_back(Block(in_channels, out_channels, stride, downsample));

    in_channels = out_channels * Block::expansion;

    for (int64_t i = 0; i < blocks; i++) {
       layers->push_back(Block(in_channels, out_channels));
    }

    return layers;
  }
};

ResNet<BasicBlock> ResNet18() {   
   ResNet<BasicBlock> model({2, 2, 2, 2}, 10);
   return model;
}

ResNet<BasicBlock> ResNet34() {
   ResNet<BasicBlock> model({3, 4, 6, 3}, 10);
   return model;
}

ResNet<BottleNeck> ResNet50() {
   ResNet<BottleNeck> model({3, 4, 6, 3}, 10);
   return model;
}

ResNet<BottleNeck> ResNet101() {
   ResNet<BottleNeck> model({3, 4, 23, 3}, 10);
   return model;
}

ResNet<BottleNeck> ResNet152() {
   ResNet<BottleNeck> model({3, 8, 36, 3}, 10);
   return model;
}

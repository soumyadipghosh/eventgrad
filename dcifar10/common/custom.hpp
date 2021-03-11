#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <stdint.h>

struct ReadOptions {
  size_t image_size = 32;
  std::string datasetPath =
           "/afs/crc.nd.edu/user/s/sghosh2/Public/CIFAR-10-images/";
};

std::map<int, std::string> labels = {
    {0, "airplane"}, {1, "automobile"}, {2, "bird"}, {3, "cat"},
    {4, "deer"},     {5, "dog"},        {6, "frog"}, {7, "horse"},
    {8, "ship"},     {9, "truck"}
};

static ReadOptions readOptions;

using Data = std::vector<std::pair<std::string, long>>;

// creating the dataset as a class
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

  using Example = torch::data::Example<>;
  Data data;

public:
  CustomDataset(const Data &data) : data(data) {}
  Example get(size_t index) {
    std::string path = readOptions.datasetPath + data[index].first;
    auto mat = cv::imread(path);

    // make sure you didn't read an empty matrix
    assert(!mat.empty());

    // make the image 224 by 224
    cv::resize(mat, mat, cv::Size(readOptions.image_size, readOptions.image_size));
    std::vector<cv::Mat> channels(3);

    // split the RGB parts of the image
    cv::split(mat, channels);

    // Channels, making each channel a torch::KUInt8 to be used
    auto R = torch::from_blob(channels[2].ptr(),
                              {readOptions.image_size, readOptions.image_size},
                              torch::kUInt8);
    auto G = torch::from_blob(channels[1].ptr(),
                              {readOptions.image_size, readOptions.image_size},
                              torch::kUInt8);
    auto B = torch::from_blob(channels[0].ptr(),
                              {readOptions.image_size, readOptions.image_size},
                              torch::kUInt8);
    auto tdata = torch::cat({R, G, B})
                     .view({3, readOptions.image_size, readOptions.image_size})
                     .to(torch::kFloat);
    auto tlabel = torch::from_blob(&data[index].second, {1}, torch::kLong);
    return {tdata, tlabel};
  } 
  torch::optional<size_t> size() const { return data.size(); }
};

std::pair<Data, Data> readInfo() {

  Data train, test;

  // std::ifstream stream(options.infoFilePath);
  // assert(stream.is_open());
  // long label;

  std::string path, type, label, totalpath;

  // make training handles
  for (int j = 0; j < 10; j++) { // 10 classes

    for (int i = 0; i < 5000; i++) { // 5000 images per class

      type = "train/";
      label = labels.at(j) + "/";

      // forming path for current image
      path = std::to_string(i);
      int l = path.length();

      // padding extra 0's
      for (int h = 0; h < (4 - l); h++) {
        path = "0" + path;
      }
      path = path + ".jpg";
      totalpath = type + label + path;
      train.push_back(std::make_pair(totalpath, j));
    }
  }

  // make testing handles
  for (int j = 0; j < 10; j++) {

    for (int i = 0; i < 1000; i++) {

      type = "test/";
      label = labels.at(j) + "/";

      // forming path for current image
      path = std::to_string(i);
      int l = path.length();

      // padding extra 0's
      for (int h = 0; h < (4 - l); h++) {
        path = "0" + path;
      }
      path = path + ".jpg";
      totalpath = type + label + path;
      test.push_back(std::make_pair(totalpath, j));
    }
  }
  std::random_shuffle(train.begin(), train.end());
  std::random_shuffle(test.begin(), test.end());
  return std::make_pair(train, test);
}


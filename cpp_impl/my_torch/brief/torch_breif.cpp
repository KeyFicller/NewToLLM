#include "my_torch.h"

#include <ATen/core/TensorBody.h>
#include <iostream>
#include <torch/data.h>
#include <torch/data/dataloader.h>
#include <torch/data/dataloader_options.h>
#include <torch/data/samplers/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/torch.h>

namespace MyTorch {

struct NeuralNetworkImpl : public torch::nn::Module {
public:
  NeuralNetworkImpl(int num_inputs, int num_outputs)
      : torch::nn::Module("NeuralNetwork") {
    layers = register_module(
        "layers", torch::nn::Sequential(
                      torch::nn::Linear(num_inputs, 30), torch::nn::ReLU(),
                      torch::nn::Linear(30, 20), torch::nn::ReLU(),
                      torch::nn::Linear(20, num_outputs)));
  }

  torch::Tensor forward(torch::Tensor x) {
    auto logits = layers->forward(x);
    return logits;
  }

public:
  torch::nn::Sequential layers{nullptr};
};

TORCH_MODULE(NeuralNetwork);

struct ToyDataset : public torch::data::Dataset<ToyDataset> {
  torch::Tensor m_features;
  torch::Tensor m_labels;

  ToyDataset(torch::Tensor features, torch::Tensor labels)
      : m_features(std::move(features)), m_labels(std::move(labels)) {}

  torch::data::Example<> get(size_t index) override {
    return {m_features[index], m_labels[index]};
  }

  torch::optional<size_t> size() const override { return m_labels.size(0); }
};

template <typename Model, typename Loader>
double compute_accuracy(Model &model, Loader &loader) {
  model->eval();
  torch::NoGradGuard no_grad;

  int correct = 0;
  int total_examples = 0;

  for (auto &batch : *loader) {
    std::vector<torch::Tensor> features;
    std::vector<torch::Tensor> labels;
    for (auto &ex : batch) {
      features.push_back(ex.data);
      labels.push_back(ex.target);
    }
    auto features_tensor = torch::stack(features).to(torch::kFloat32);
    auto labels_tensor = torch::stack(labels).to(torch::kLong);

    auto logits = model->forward(features_tensor);
    auto predictions = torch::argmax(logits, 1);

    auto compare = labels_tensor.eq(predictions);
    correct += compare.sum().template item<int64_t>();
    total_examples += features.size();
  }

  return 1.0 * correct / total_examples;
}

void brief_torch() {
  // Scalar/Vector/Matrix/Tensor
  torch::Tensor tensor0d = torch::tensor(1);
  std::cout << "tensor0d: " << tensor0d << std::endl;
  torch::Tensor tensor1d = torch::tensor({1, 2, 3});
  std::cout << "tensor1d: " << tensor1d << std::endl;
  torch::Tensor tensor2d = torch::tensor({{1, 2, 3}, {4, 5, 6}});
  std::cout << "tensor2d: " << tensor2d << std::endl;
  torch::Tensor tensor3d =
      torch::tensor({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
  std::cout << "tensor3d: " << tensor3d << std::endl;

  // Tensor data type
  torch::Tensor tensorint = torch::tensor({1, 2, 3});
  std::cout << "tensorint.dtype: " << tensorint.dtype() << std::endl;
  torch::Tensor tensorfloat = torch::tensor({1.0, 2.0, 3.0});
  std::cout << "tensorfloat.dtype: " << tensorfloat.dtype() << std::endl;
  torch::Tensor tensorfloat64 = tensorfloat.to(torch::kFloat64);
  std::cout << "tensorfloat64.dtype: " << tensorfloat64.dtype() << std::endl;

  // Basic operations
  std::cout << "tensor2d: " << tensor2d << std::endl;
  std::cout << "tensor2d.reshape: "
            << tensor2d.reshape({tensor2d.size(1), tensor2d.size(0)})
            << std::endl;
  std::cout << "tensor2d.view: "
            << tensor2d.view({tensor2d.size(1), tensor2d.size(0)}) << std::endl;
  std::cout << "tensor2d.T: " << tensor2d.transpose(0, 1) << std::endl;

  std::cout << "tensor2d.matmul(tensor2d.T): "
            << tensor2d.matmul(tensor2d.transpose(0, 1)) << std::endl;
  std::cout << "tensor2d @ tensor2d.T: "
            << tensor2d.matmul(tensor2d.transpose(0, 1)) << std::endl;

  // Auto grad
  torch::Tensor a = torch::tensor({1.0});
  torch::Tensor b = torch::tensor({2.0});
  torch::Tensor x = torch::tensor({3.0}, torch::requires_grad(true));

  torch::Tensor y = a * x + b;
  std::cout << "y = a * x + b, a = 1, grad(y, x) "
            << torch::autograd::grad({y}, {x}, {}, true) << std::endl;

  y.backward();
  std::cout << "x.grad: " << x.grad() << std::endl;

  // Neural network
  torch::manual_seed(1234);
  NeuralNetwork model(50, 3);
  std::cout << "neural network: " << model << std::endl;
  int num_params = 0;
  for (const auto &param : model->parameters()) {
    num_params += param.numel();
  }
  std::cout << "number of parameters: " << num_params << std::endl;

  auto layer0_any = model->layers->ptr(0);
  auto layer0 = std::dynamic_pointer_cast<torch::nn::LinearImpl>(layer0_any);
  if (layer0) {
    std::cout << "weights for layer[0]: " << layer0->weight << std::endl;
    std::cout << "shape of weights for layer[0]: " << layer0->weight.sizes()
              << std::endl;
  }

  torch::Tensor X = torch::rand({1, 50});
  std::cout << "Neural network input: " << X << std::endl;
  auto out = model->forward(X);
  std::cout << "Neural network output with grad: " << out << std::endl;

  {
    torch::NoGradGuard no_grad;
    out = torch::softmax(model->forward(X), 1);
  }
  std::cout << "Neural network output without grad: " << out << std::endl;

  // Data loading and preprocessing
  torch::Tensor X_train = torch::tensor(
      {{-1.2, 3.1}, {-0.9, 2.9}, {-0.5, 2.6}, {2.3, -1.1}, {2.7, -1.5}});
  torch::Tensor y_train = torch::tensor({0, 0, 0, 1, 1});
  torch::Tensor X_test = torch::tensor({{-0.8, 2.8}, {2.6, -1.6}});
  torch::Tensor y_test = torch::tensor({0, 1});
  ToyDataset train_ds(X_train, y_train);
  ToyDataset test_ds(X_test, y_test);
  std::cout << "len of train_ds: " << train_ds.size().value() << std::endl;
  std::cout << "len of test_ds: " << test_ds.size().value() << std::endl;

  torch::manual_seed(1234);
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(train_ds),
          torch::data::DataLoaderOptions().batch_size(2).drop_last(true));
  auto test_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(test_ds),
          torch::data::DataLoaderOptions().batch_size(2).drop_last(true));
  size_t idx = 0;
  for (auto &batch : *train_loader) {
    std::cout << "Batch " << idx << ", size = " << batch.size() << std::endl;
    for (auto &ex : batch) {
      std::cout << "Example " << idx << ": x = " << ex.data
                << ", y = " << ex.target << std::endl;
    }
    ++idx;
  }

  for (auto &batch : *test_loader) {
    std::cout << "Batch " << idx << ", size = " << batch.size() << std::endl;
    for (auto &ex : batch) {
      std::cout << "Example " << idx << ": x = " << ex.data
                << ", y = " << ex.target << std::endl;
    }
    ++idx;
  }

  torch::manual_seed(123);
  NeuralNetwork example_model(2, 2);
  auto optimizer = torch::optim::SGD(example_model->parameters(), 0.5);
  int num_epochs = 4;
  int batch_idx = 0;
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    example_model->train();

    for (auto &batch : *train_loader) {
      std::vector<torch::Tensor> features;
      std::vector<torch::Tensor> labels;
      for (auto &ex : batch) {
        features.push_back(ex.data);
        labels.push_back(ex.target);
      }
      auto features_tensor = torch::stack(features).to(torch::kFloat32);
      auto labels_tensor = torch::stack(labels).to(torch::kLong);
      auto logits = example_model->forward(features_tensor);
      auto loss = torch::nn::functional::cross_entropy(logits, labels_tensor);

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();

      std::cout << "Epoch: " << epoch << ", Batch: " << batch_idx
                << ", Loss: " << std::fixed << std::setprecision(2)
                << loss.item<float>() << std::endl;
      ++batch_idx;
    }
  }

  example_model->eval();
  torch::Tensor outputs;
  {
    torch::NoGradGuard no_grad;
    outputs = example_model->forward(X_train.to(torch::kFloat32));
    std::cout << "Training outputs: " << outputs << std::endl;
  }
  auto proabs = torch::softmax(outputs, 1);
  std::cout << "Probabilities: " << proabs << std::endl;
  auto predictions = torch::argmax(proabs, 1);
  std::cout << "Predictions: " << predictions << std::endl;
  std::cout << "Training accuracy: "
            << compute_accuracy(example_model, train_loader) << std::endl;
  std::cout << "Test accuracy: " << compute_accuracy(example_model, test_loader)
            << std::endl;
}
} // namespace MyTorch
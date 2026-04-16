#include "my_torch.h"

#include <iostream>
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
}
} // namespace MyTorch
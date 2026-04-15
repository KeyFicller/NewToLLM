#include "my_torch.h"

#include <iostream>
#include <torch/torch.h>

namespace MyTorch {
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
}
} // namespace MyTorch
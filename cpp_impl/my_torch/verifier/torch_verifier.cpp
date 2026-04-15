#include "my_torch.h"
#include <iostream>
#include <torch/version.h>

namespace MyTorch {
void verify_torch() {
  std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
}
} // namespace MyTorch
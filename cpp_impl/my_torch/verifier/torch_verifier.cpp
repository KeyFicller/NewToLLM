#include "my_torch.h"
#include "utils/torch_utils.h"
#include <ATen/Context.h>
#include <c10/core/Backend.h>
#include <c10/core/Device.h>
#include <iostream>
#include <torch/cuda.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/version.h>

namespace MyTorch {

void verify_torch() {
  // Verify version
  std::cout << "[verify] Torch version is: " << TORCH_VERSION << '\n';

  // Verify device
  std::cout << "[verify] Torch device should be: "
            << c10::DeviceTypeName(decl_device().type(), true) << '\n';
}
} // namespace MyTorch
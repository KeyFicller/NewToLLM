#include "torch_utils.h"
#include "utils/runtime.h"

namespace MyTorch {

torch::Device decl_device() {
  if (!runtime_options().device.empty()) {
    return torch::Device(runtime_options().device);
  }
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
  } else if (torch::hasMPS()) {
    device = torch::Device(torch::kMPS);
  }
  return device;
}

} // namespace MyTorch

#pragma once

#include <c10/core/Device.h>
#include <torch/torch.h>

namespace MyTorch {

torch::Device decl_device();

} // namespace MyTorch

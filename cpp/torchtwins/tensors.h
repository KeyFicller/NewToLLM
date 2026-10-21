#pragma once
// Lesson 01 - tensors (twin of py/torchtwins/tensors.py).

#include "torchtwins/config.h"

#include <torch/torch.h>

#include <cstdint>

namespace torchtwins {

[[nodiscard]] torch::Tensor create(uint64_t seed);
void reshape(const torch::Tensor& x);
void index(const torch::Tensor& x);
void broadcast(const torch::Tensor& x);
void matmul(const torch::Tensor& x);
void run_tensors(const Preset& preset);

}  // namespace torchtwins

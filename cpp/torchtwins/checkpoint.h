#pragma once
// Lesson 08 - checkpoints (twin of py/torchtwins/checkpoint.py).

#include "torchtwins/config.h"
#include "torchtwins/gpt.h"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>

namespace torchtwins {

void save_weights(GPTModel& model, const std::filesystem::path& path);
void load_weights(GPTModel& model, const std::filesystem::path& path);
[[nodiscard]] int64_t tensor_count(GPTModel& model);
[[nodiscard]] double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b);
void run_checkpoint(const Preset& preset);

}  // namespace torchtwins

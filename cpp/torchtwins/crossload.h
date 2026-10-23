#pragma once
// Lesson 09 - portable weight bridge (twin of py/torchtwins/crossload.py).

#include "torchtwins/config.h"
#include "torchtwins/gpt.h"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace torchtwins {

void export_portable(GPTModel& model, const std::filesystem::path& path);
void import_portable(GPTModel& model, const std::filesystem::path& path);
[[nodiscard]] std::vector<std::string> param_names(GPTModel& model);
[[nodiscard]] double logits_diff(const torch::Tensor& a, const torch::Tensor& b);
void run_crossload(const Preset& preset);

}  // namespace torchtwins

#pragma once
// Lesson 06a - data pipeline (twin of py/newtollm/data.py).

#include "newtollm/config.h"
#include "newtollm/tokenizer.h"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <utility>
#include <vector>

namespace newtollm {

[[nodiscard]] std::filesystem::path default_text_path();
[[nodiscard]] std::string read_text(const std::filesystem::path& path);

// @twin Windows
struct Windows {
  torch::Tensor inputs;   // [num_windows, context_length] int64
  torch::Tensor targets;  // next tokens (same shape) - or class labels [n] in lesson 07

  [[nodiscard]] int64_t size() const { return inputs.size(0); }
};

using Batch = std::pair<torch::Tensor, torch::Tensor>;

[[nodiscard]] std::pair<torch::Tensor, torch::Tensor> split_ids(const torch::Tensor& ids,
                                                                double train_ratio);
[[nodiscard]] Windows make_windows(const torch::Tensor& ids, int64_t context_length,
                                   int64_t stride);
// Python yields lazily; a vector of views is the idiomatic eager C++ equivalent.
[[nodiscard]] std::vector<Batch> batches(
    const Windows& windows, int64_t batch_size,
    const std::optional<at::Generator>& generator = std::nullopt);

// @twin Dataset
struct Dataset {
  Windows train;
  Windows val;
};

[[nodiscard]] Dataset load_dataset(const Preset& preset, const Tokenizer& tokenizer,
                                   const std::filesystem::path& path = default_text_path());
void run_data(const Preset& preset);

}  // namespace newtollm

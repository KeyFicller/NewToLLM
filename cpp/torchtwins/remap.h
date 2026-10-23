#pragma once
// Lesson 10 - OpenAI / HuggingFace GPT-2 key remapping (twin of py/torchtwins/remap.py).

#include "torchtwins/config.h"

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace torchtwins {

[[nodiscard]] torch::Tensor conv1d_to_linear(const torch::Tensor& weight);
[[nodiscard]] torch::Tensor linear_to_conv1d(const torch::Tensor& weight);
[[nodiscard]] std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_qkv(
    const torch::Tensor& combined, int64_t emb_dim);
[[nodiscard]] torch::Tensor merge_qkv(const torch::Tensor& query, const torch::Tensor& key,
                                      const torch::Tensor& value);
[[nodiscard]] std::unordered_map<std::string, torch::Tensor> ours_to_hf(
    const std::unordered_map<std::string, torch::Tensor>& ours, const GPTConfig& cfg);
[[nodiscard]] std::unordered_map<std::string, torch::Tensor> hf_to_ours(
    const std::unordered_map<std::string, torch::Tensor>& hf, const GPTConfig& cfg);
void apply_state(torch::nn::Module& model,
                 const std::unordered_map<std::string, torch::Tensor>& state);
void run_remap(const Preset& preset);

}  // namespace torchtwins

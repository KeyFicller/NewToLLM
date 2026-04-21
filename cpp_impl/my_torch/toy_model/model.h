#pragma once

#include "toy_model/config.h"
#include "toy_model/transformer.h"

#include <torch/torch.h>

namespace MyTorch {

// Same Impl + TORCH_MODULE pattern as transformer components.
struct ToyModelImpl : torch::nn::Module {
  explicit ToyModelImpl(const ToyModelConfig& cfg);
  torch::Tensor forward(const torch::Tensor& in_idx);

  // Keep a copy of config in C++ object state.
  ToyModelConfig cfg_;
  torch::nn::Embedding tok_emb{nullptr};
  torch::nn::Embedding pos_emb{nullptr};
  torch::nn::Dropout drop_emb{nullptr};
  torch::nn::Sequential trf_blocks{nullptr};
  ToyLayerNorm final_norm{nullptr};
  torch::nn::Linear out_head{nullptr};
};
TORCH_MODULE(ToyModel);

torch::Tensor generate_text_simple(
    ToyModel& model,
    torch::Tensor idx,
    int64_t max_new_tokens,
    int64_t context_length);

} // namespace MyTorch

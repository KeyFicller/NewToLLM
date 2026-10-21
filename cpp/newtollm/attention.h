#pragma once
// Lesson 04 - causal multi-head self-attention (twin of py/newtollm/attention.py).

#include "newtollm/config.h"

#include <torch/torch.h>

#include <cstdint>

namespace newtollm {

[[nodiscard]] torch::Tensor causal_mask(int64_t context_length);

// @twin MultiHeadAttention
class MultiHeadAttentionImpl : public torch::nn::Module {
 public:
  MultiHeadAttentionImpl(int64_t d_in, int64_t d_out, int64_t context_length, double drop_rate,
                         int64_t n_heads, bool qkv_bias = false);

  [[nodiscard]] torch::Tensor split_heads(const torch::Tensor& x) const;
  [[nodiscard]] torch::Tensor attention_weights(const torch::Tensor& x);
  torch::Tensor forward(const torch::Tensor& x);

 private:
  // Members are initialised in *declaration* order (not initialiser-list order),
  // so this order must match the Python __init__ to draw identical random weights.
  int64_t d_out_;
  int64_t n_heads_;
  int64_t head_dim_;
  torch::nn::Linear w_query_;
  torch::nn::Linear w_key_;
  torch::nn::Linear w_value_;
  torch::nn::Linear out_proj_;
  torch::nn::Dropout dropout_;
  torch::Tensor mask_;
};
TORCH_MODULE(MultiHeadAttention);

void run_attention(const Preset& preset);

}  // namespace newtollm

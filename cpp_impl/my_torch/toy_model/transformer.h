#pragma once

#include "toy_model/config.h"

#include <torch/torch.h>

namespace MyTorch {

// In LibTorch, custom modules are commonly declared as `FooImpl` + `TORCH_MODULE(Foo)`.
// This differs from Python class usage and gives shared_ptr-like ownership semantics.
struct ToyMultiHeadAttentionImpl : torch::nn::Module {
  ToyMultiHeadAttentionImpl(
      int64_t d_in,
      int64_t d_out,
      int64_t context_length,
      double drop_rate,
      int64_t n_heads,
      bool qkv_bias);

  torch::Tensor forward(const torch::Tensor& x);

  // Keep scalar metadata as plain C++ fields.
  int64_t d_out_;
  int64_t n_heads_;
  int64_t d_head_;
  // Every submodule should be initialized with `{nullptr}` and later registered.
  torch::nn::Linear W_query{nullptr};
  torch::nn::Linear W_key{nullptr};
  torch::nn::Linear W_value{nullptr};
  torch::nn::Linear out_proj{nullptr};
  torch::nn::Dropout drop_out{nullptr};
};
TORCH_MODULE(ToyMultiHeadAttention);

struct ToyFeedForwardImpl : torch::nn::Module {
  explicit ToyFeedForwardImpl(const ToyModelConfig& cfg);
  torch::Tensor forward(const torch::Tensor& x);
  torch::nn::Sequential layers{nullptr};
};
TORCH_MODULE(ToyFeedForward);

struct ToyLayerNormImpl : torch::nn::Module {
  explicit ToyLayerNormImpl(int64_t emb_dim);
  torch::Tensor forward(const torch::Tensor& x);

  // Parameters are explicit tensors in C++, unlike Python where attribute assignment registers them.
  double eps = 1e-5;
  torch::Tensor scale;
  torch::Tensor bias;
};
TORCH_MODULE(ToyLayerNorm);

struct ToyTransformerBlockImpl : torch::nn::Module {
  explicit ToyTransformerBlockImpl(const ToyModelConfig& cfg);
  torch::Tensor forward(const torch::Tensor& x_in);

  ToyMultiHeadAttention att{nullptr};
  ToyFeedForward ff{nullptr};
  ToyLayerNorm norm1{nullptr};
  ToyLayerNorm norm2{nullptr};
  torch::nn::Dropout drop_out{nullptr};
};
TORCH_MODULE(ToyTransformerBlock);

} // namespace MyTorch

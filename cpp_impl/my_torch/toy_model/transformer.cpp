#include "toy_model/transformer.h"

#include <cmath>
#include <limits>

namespace MyTorch {
// LibTorch slicing uses torch::indexing utilities instead of Python-style ":" syntax.
using torch::indexing::None;
using torch::indexing::Slice;

ToyMultiHeadAttentionImpl::ToyMultiHeadAttentionImpl(
    int64_t d_in,
    int64_t d_out,
    int64_t context_length,
    double drop_rate,
    int64_t n_heads,
    bool qkv_bias)
    : d_out_(d_out),
      n_heads_(n_heads),
      d_head_(d_out / n_heads),
      w_query(register_module(
          "w_query",
          torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      w_key(register_module(
          "w_key",
          torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      w_value(register_module(
          "w_value",
          torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      out_proj(register_module("out_proj", torch::nn::Linear(d_out, d_out))),
      dropout(register_module("dropout", torch::nn::Dropout(drop_rate))) {
  TORCH_CHECK(d_out % n_heads == 0, "d_out must be divisible by n_heads");
  auto mask =
      torch::triu(torch::ones({context_length, context_length}, torch::kFloat32), 1);
  // Buffers (e.g. causal mask) must be registered explicitly to move with .to(device)
  // and to be included in state dict serialization.
  register_buffer("mask", mask);
}

torch::Tensor ToyMultiHeadAttentionImpl::forward(const torch::Tensor& x) {
  const auto batch_size = x.size(0);
  const auto n_tokens = x.size(1);

  auto keys = w_key(x);
  auto queries = w_query(x);
  auto values = w_value(x);

  keys = keys.view({batch_size, n_tokens, n_heads_, d_head_}).transpose(1, 2);
  queries = queries.view({batch_size, n_tokens, n_heads_, d_head_}).transpose(1, 2);
  values = values.view({batch_size, n_tokens, n_heads_, d_head_}).transpose(1, 2);

  auto attn_scores = torch::matmul(queries, keys.transpose(2, 3));
  // Access registered buffer by name; there is no direct Python-like attribute fallback.
  auto mask_bool = this->named_buffers()["mask"]
                       .to(torch::kBool)
                       .index({Slice(None, n_tokens), Slice(None, n_tokens)});
  attn_scores.masked_fill_(mask_bool, -std::numeric_limits<float>::infinity());

  auto scale = std::sqrt(static_cast<double>(keys.size(-1)));
  auto attn_weights = torch::softmax(attn_scores / scale, -1);
  attn_weights = dropout(attn_weights);

  auto context = torch::matmul(attn_weights, values).transpose(1, 2);
  context = context.contiguous().view({batch_size, n_tokens, d_out_});
  return out_proj(context);
}

ToyFeedForwardImpl::ToyFeedForwardImpl(const ToyModelConfig& cfg)
    : layers(register_module(
          "layers",
          torch::nn::Sequential(
              torch::nn::Linear(cfg.emb_dim, cfg.emb_dim * 4),
              torch::nn::GELU(),
              torch::nn::Linear(cfg.emb_dim * 4, cfg.emb_dim)))) {}

torch::Tensor ToyFeedForwardImpl::forward(const torch::Tensor& x) {
  // For nn::Sequential in C++, use typed forward<> to avoid void return deduction.
  return layers->forward<torch::Tensor>(x);
}

ToyLayerNormImpl::ToyLayerNormImpl(int64_t emb_dim)
    // In C++, learnable tensors are created and registered with register_parameter.
    : scale(register_parameter("scale", torch::ones({emb_dim}))),
      bias(register_parameter("bias", torch::zeros({emb_dim}))) {}

torch::Tensor ToyLayerNormImpl::forward(const torch::Tensor& x) {
  auto mean = x.mean(-1, true);
  auto var = x.var(-1, false, true);
  auto norm = (x - mean) / torch::sqrt(var + eps);
  return scale * norm + bias;
}

ToyTransformerBlockImpl::ToyTransformerBlockImpl(const ToyModelConfig& cfg)
    : att(register_module(
          "att",
          ToyMultiHeadAttention(
              cfg.emb_dim,
              cfg.emb_dim,
              cfg.context_length,
              cfg.drop_rate,
              cfg.n_heads,
              cfg.qkv_bias))),
      ff(register_module("ff", ToyFeedForward(cfg))),
      norm1(register_module("norm1", ToyLayerNorm(cfg.emb_dim))),
      norm2(register_module("norm2", ToyLayerNorm(cfg.emb_dim))),
      dropout(register_module("dropout", torch::nn::Dropout(cfg.drop_rate))) {}

torch::Tensor ToyTransformerBlockImpl::forward(const torch::Tensor& x_in) {
  auto x = x_in;
  auto shortcut = x;
  x = norm1(x);
  x = att(x);
  x = dropout(x);
  x = x + shortcut;

  shortcut = x;
  x = norm2(x);
  x = ff(x);
  x = dropout(x);
  x = x + shortcut;
  return x;
}

} // namespace MyTorch

// Lesson 04 - causal multi-head self-attention.
#include "newtollm/attention.h"

#include "newtollm/report.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace newtollm {

using torch::indexing::None;
using torch::indexing::Slice;

// @twin causal_mask
torch::Tensor causal_mask(int64_t context_length) {
  // True above the diagonal: position t may not look at t+1, t+2, ...
  return torch::triu(torch::ones({context_length, context_length}, torch::kBool), /*diagonal=*/1);
}

namespace {
int64_t checked_head_dim(int64_t d_out, int64_t n_heads) {
  if (d_out % n_heads != 0) {
    throw std::invalid_argument("d_out must be divisible by n_heads");
  }
  return d_out / n_heads;
}
}  // namespace

// @twin MultiHeadAttention
MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t d_in, int64_t d_out, int64_t context_length,
                                               double drop_rate, int64_t n_heads, bool qkv_bias)
    : d_out_(d_out),
      n_heads_(n_heads),
      head_dim_(checked_head_dim(d_out, n_heads)),
      w_query_(register_module(
          "w_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      w_key_(register_module(
          "w_key", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      w_value_(register_module(
          "w_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      out_proj_(register_module("out_proj", torch::nn::Linear(d_out, d_out))),
      dropout_(register_module("dropout", torch::nn::Dropout(drop_rate))),
      mask_(register_buffer("mask", causal_mask(context_length))) {}

torch::Tensor MultiHeadAttentionImpl::split_heads(const torch::Tensor& x) const {
  // [batch, tokens, d_out] -> [batch, heads, tokens, head_dim]
  return x.view({x.size(0), x.size(1), n_heads_, head_dim_}).transpose(1, 2);
}

torch::Tensor MultiHeadAttentionImpl::attention_weights(const torch::Tensor& x) {
  const auto tokens = x.size(1);
  const auto queries = split_heads(w_query_(x));
  const auto keys = split_heads(w_key_(x));
  auto scores = torch::matmul(queries, keys.transpose(2, 3));
  scores = scores.masked_fill(mask_.index({Slice(None, tokens), Slice(None, tokens)}),
                              -std::numeric_limits<float>::infinity());
  return torch::softmax(scores / std::sqrt(static_cast<double>(head_dim_)), /*dim=*/-1);
}

torch::Tensor MultiHeadAttentionImpl::forward(const torch::Tensor& x) {
  const auto batch = x.size(0);
  const auto tokens = x.size(1);
  const auto weights = dropout_(attention_weights(x));
  auto context = torch::matmul(weights, split_heads(w_value_(x)));
  context = context.transpose(1, 2).contiguous().view({batch, tokens, d_out_});
  return out_proj_(context);
}

// @twin run_attention
void run_attention(const Preset& preset) {
  const auto& cfg = preset.model;
  torch::manual_seed(preset.seed);
  const auto x = torch::randn({1, 4, cfg.emb_dim});

  report::section("causal_mask");
  report::show("causal_mask(4)", causal_mask(4));

  report::section("MultiHeadAttention");
  MultiHeadAttention attention(cfg.emb_dim, cfg.emb_dim, cfg.context_length, 0.0, cfg.n_heads,
                               cfg.qkv_bias);
  const auto weights = attention->attention_weights(x);
  report::show("weights[0, 0]", weights.index({0, 0}));
  report::show("weights.sum(-1)", weights.sum(/*dim=*/-1));
  report::show("attention(x)", attention(x));
}

}  // namespace newtollm

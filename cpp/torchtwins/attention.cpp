// Lesson 04 - causal multi-head self-attention.
#include "torchtwins/attention.h"

#include "torchtwins/report.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace torchtwins {

using torch::indexing::None;
using torch::indexing::Slice;

// @twin causal_mask
torch::Tensor causal_mask(int64_t context_length) {
  // @step triu
  // True above the diagonal: position t may not look at t+1, t+2, ...
  // Row t = query position, column s = key position. diagonal=1 marks every s > t,
  // i.e. the future. Those cells get -inf before softmax, and exp(-inf) = 0, so each
  // token's weights are spread only over itself and earlier tokens. That is what
  // lets one forward pass train every position at once without peeking ahead.
  return torch::triu(torch::ones({context_length, context_length}, torch::kBool), /*diagonal=*/1);
}

// @twin MultiHeadAttention
// @step check
namespace {
int64_t checked_head_dim(int64_t d_out, int64_t n_heads) {
  if (d_out % n_heads != 0) {
    throw std::invalid_argument("d_out must be divisible by n_heads");
  }
  return d_out / n_heads;
}
}  // namespace

// @step layers
MultiHeadAttentionImpl::MultiHeadAttentionImpl(int64_t d_in, int64_t d_out, int64_t context_length,
                                               double drop_rate, int64_t n_heads, bool qkv_bias)
    : d_out_(d_out),
      n_heads_(n_heads),
      head_dim_(checked_head_dim(d_out, n_heads)),
      // register_module() records the child under a name (for parameters(), to(), save)
      // and returns the same holder, which we also keep as a member for direct calls.
      w_query_(register_module(
          "w_query", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      w_key_(register_module(
          "w_key", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      w_value_(register_module(
          "w_value", torch::nn::Linear(torch::nn::LinearOptions(d_in, d_out).bias(qkv_bias)))),
      out_proj_(register_module("out_proj", torch::nn::Linear(d_out, d_out))),
      dropout_(register_module("dropout", torch::nn::Dropout(drop_rate))),
      // A buffer, not a Parameter: it moves with ->to(device) and shows up in buffers(),
      // but the optimizer never sees it. LibTorch has no persistent=false flag.
      mask_(register_buffer("mask", causal_mask(context_length))) {}

// @step split
torch::Tensor MultiHeadAttentionImpl::split_heads(const torch::Tensor& x) const {
  // [batch, tokens, d_out] -> [batch, heads, tokens, head_dim]
  // (batch, tokens, d_out) -> view (batch, tokens, heads, head_dim)
  // -> transpose (batch, heads, tokens, head_dim): heads become a batch-like dim,
  // so one matmul below attends in every head in parallel.
  return x.view({x.size(0), x.size(1), n_heads_, head_dim_}).transpose(1, 2);
}

// @step qk
torch::Tensor MultiHeadAttentionImpl::attention_weights(const torch::Tensor& x) {
  const auto tokens = x.size(1);
  const auto queries = split_heads(w_query_(x));
  const auto keys = split_heads(w_key_(x));
  // (b, h, T, hd) @ (b, h, hd, T) -> (b, h, T, T): score of every query vs every key.
  // C++ has no @ operator for tensors; torch::matmul is the same op.
  auto scores = torch::matmul(queries, keys.transpose(2, 3));
  // @step mask
  // The buffer is sized for context_length; crop it to the actual sequence length.
  // Python's mask[:T, :T] is spelled index({Slice(None, T), Slice(None, T)}).
  scores = scores.masked_fill(mask_.index({Slice(None, tokens), Slice(None, tokens)}),
                              -std::numeric_limits<float>::infinity());
  // @step softmax
  // Dividing by sqrt(head_dim) keeps the dot products' variance near 1, so softmax
  // does not saturate into a one-hot and gradients keep flowing. dim=-1 = over keys.
  return torch::softmax(scores / std::sqrt(static_cast<double>(head_dim_)), /*dim=*/-1);
}

// @step mix
torch::Tensor MultiHeadAttentionImpl::forward(const torch::Tensor& x) {
  const auto batch = x.size(0);
  const auto tokens = x.size(1);
  const auto weights = dropout_(attention_weights(x));
  auto context = torch::matmul(weights, split_heads(w_value_(x)));
  // @step merge
  // (b, h, T, hd) -> (b, T, h, hd) -> (b, T, d_out). transpose() leaves memory
  // non-contiguous, so contiguous() copies it before view() can merge the heads.
  context = context.transpose(1, 2).contiguous().view({batch, tokens, d_out_});
  return out_proj_(context);
}

// @twin run_attention
void run_attention(const Preset& preset) {
  // @step input
  const auto& cfg = preset.model;
  torch::manual_seed(preset.seed);
  const auto x = torch::randn({1, 4, cfg.emb_dim});

  // @step mask
  report::section("causal_mask");
  report::show("causal_mask(4)", causal_mask(4));

  // @step attn
  report::section("MultiHeadAttention");
  MultiHeadAttention attention(cfg.emb_dim, cfg.emb_dim, cfg.context_length, 0.0, cfg.n_heads,
                               cfg.qkv_bias);
  // `->` because `attention` is the TORCH_MODULE holder; the method lives on the Impl.
  const auto weights = attention->attention_weights(x);
  report::show("weights[0, 0]", weights.index({0, 0}));
  report::show("weights.sum(-1)", weights.sum(/*dim=*/-1));
  report::show("attention(x)", attention(x));
}

}  // namespace torchtwins

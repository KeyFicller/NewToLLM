#include "toy_model/model.h"

#include <algorithm>
#include <limits>

namespace MyTorch {
// LibTorch uses indexing helpers instead of direct Python slicing syntax.
using torch::indexing::None;
using torch::indexing::Slice;

ToyModelImpl::ToyModelImpl(const ToyModelConfig& cfg)
    : cfg_(cfg),
      tok_emb(register_module(
          "tok_emb",
          torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim))),
      pos_emb(register_module(
          "pos_emb",
          torch::nn::Embedding(cfg.context_length, cfg.emb_dim))),
      drop_emb(register_module("drop_emb", torch::nn::Dropout(cfg.drop_rate))),
      trf_blocks(register_module("trf_blocks", torch::nn::Sequential())),
      final_norm(register_module("final_norm", ToyLayerNorm(cfg.emb_dim))),
      out_head(register_module(
          "out_head",
          torch::nn::Linear(
              torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size).bias(false)))) {
  // In C++, blocks are pushed one-by-one into Sequential.
  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    trf_blocks->push_back(ToyTransformerBlock(cfg));
  }
}

torch::Tensor ToyModelImpl::forward(const torch::Tensor& in_idx) {
  const auto n_tokens = in_idx.size(1);
  auto tok_embeds = tok_emb(in_idx);
  // Keep position ids on the same device as input by reusing input options and forcing Long dtype.
  auto pos_ids = torch::arange(n_tokens, in_idx.options().dtype(torch::kLong));
  auto pos_embeds = pos_emb(pos_ids);

  auto x = tok_embeds + pos_embeds;
  x = drop_emb(x);
  // Typed forward<> is important for Sequential in LibTorch.
  x = trf_blocks->forward<torch::Tensor>(x);
  x = final_norm(x);
  return out_head(x);
}

torch::Tensor generate_text_simple(
    ToyModel& model,
    torch::Tensor idx,
    int64_t max_new_tokens,
    int64_t context_length) {
  for (int64_t i = 0; i < max_new_tokens; ++i) {
    // Equivalent to Python idx[:, -context_length:].
    const auto start = std::max<int64_t>(0, idx.size(1) - context_length);
    auto idx_cond = idx.index({Slice(), Slice(start, None)});
    auto logits = model->forward(idx_cond);
    logits = logits.index({Slice(), -1, Slice()});
    auto probs = torch::softmax(logits, -1);
    auto idx_next = std::get<1>(probs.max(-1, true));
    idx = torch::cat({idx, idx_next}, 1);
  }
  return idx;
}

torch::Tensor generate_text_advanced(
    ToyModel& model,
    torch::Tensor idx,
    int64_t max_new_tokens,
    int64_t context_length,
    double temperature,
    c10::optional<int64_t> top_k,
    c10::optional<int64_t> eos_id) {
  for (int64_t i = 0; i < max_new_tokens; ++i) {
    const auto start = std::max<int64_t>(0, idx.size(1) - context_length);
    auto idx_cond = idx.index({Slice(), Slice(start, None)});
    auto logits = model->forward(idx_cond);
    logits = logits.index({Slice(), -1, Slice()});

    if (top_k.has_value()) {
      auto top_logits = std::get<0>(torch::topk(logits, *top_k, -1));
      auto min_vals = top_logits.index({Slice(), top_logits.size(1) - 1}).unsqueeze(1);
      logits = torch::where(
          logits < min_vals,
          torch::full_like(logits, -std::numeric_limits<float>::infinity()),
          logits);
    }

    torch::Tensor idx_next;
    if (temperature > 0.0) {
      auto probs = torch::softmax(logits / temperature, -1);
      idx_next = torch::multinomial(probs, 1);
    } else {
      idx_next = std::get<1>(logits.max(-1, true));
    }

    if (eos_id.has_value()) {
      auto all_eos = idx_next.eq(*eos_id).all().item<bool>();
      if (all_eos) {
        break;
      }
    }

    idx = torch::cat({idx, idx_next}, 1);
  }
  return idx;
}

} // namespace MyTorch

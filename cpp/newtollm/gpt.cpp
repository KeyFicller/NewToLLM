// Lesson 05 - the GPT model: Transformer blocks, logits, greedy generation.
#include "newtollm/gpt.h"

#include "newtollm/report.h"
#include "newtollm/tokenizer.h"

#include <algorithm>

namespace newtollm {

using torch::indexing::None;
using torch::indexing::Slice;

// @twin TransformerBlock
// @step layers
TransformerBlockImpl::TransformerBlockImpl(const GPTConfig& cfg)
    : att_(register_module("att", MultiHeadAttention(cfg.emb_dim, cfg.emb_dim, cfg.context_length,
                                                     cfg.drop_rate, cfg.n_heads, cfg.qkv_bias))),
      ff_(register_module("ff", FeedForward(cfg.emb_dim))),
      norm1_(register_module("norm1", LayerNorm(cfg.emb_dim))),
      norm2_(register_module("norm2", LayerNorm(cfg.emb_dim))),
      drop_shortcut_(register_module("drop_shortcut", torch::nn::Dropout(cfg.drop_rate))) {}

// @step residual
torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
  x = x + drop_shortcut_(att_(norm1_(x)));
  return x + drop_shortcut_(ff_(norm2_(x)));
}

// @twin GPTModel
// @step blocks
namespace {
// Built inside the member-initialiser list (not in the constructor body) so the
// blocks draw their random init *before* final_norm/out_head, exactly like Python.
torch::nn::Sequential make_blocks(const GPTConfig& cfg) {
  torch::nn::Sequential blocks;
  for (int64_t i = 0; i < cfg.n_layers; ++i) {
    blocks->push_back(TransformerBlock(cfg));
  }
  return blocks;
}
}  // namespace

// @step layers
GPTModelImpl::GPTModelImpl(const GPTConfig& cfg)
    : tok_emb_(register_module("tok_emb", torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim))),
      pos_emb_(register_module("pos_emb", torch::nn::Embedding(cfg.context_length, cfg.emb_dim))),
      drop_emb_(register_module("drop_emb", torch::nn::Dropout(cfg.drop_rate))),
      trf_blocks_(register_module("trf_blocks", make_blocks(cfg))),
      final_norm_(register_module("final_norm", LayerNorm(cfg.emb_dim))),
      out_head_(register_module(
          "out_head",
          torch::nn::Linear(torch::nn::LinearOptions(cfg.emb_dim, cfg.vocab_size).bias(false)))) {}

// @step embed
torch::Tensor GPTModelImpl::forward(const torch::Tensor& idx) {
  const auto tokens = idx.size(1);
  const auto positions = torch::arange(tokens, idx.options().dtype(torch::kLong));
  auto x = drop_emb_(tok_emb_(idx) + pos_emb_(positions));
  // @step stack
  x = final_norm_(trf_blocks_->forward<torch::Tensor>(x));
  return out_head_(x);
}

// @step head
void GPTModelImpl::replace_head(int64_t num_outputs) {
  // replace_module() swaps the registered child; the member must be re-pointed too,
  // otherwise forward() would keep calling the old head.
  out_head_ =
      replace_module("out_head", torch::nn::Linear(out_head_->options.in_features(), num_outputs));
}

// @twin generate_greedy
torch::Tensor generate_greedy(GPTModel& model, torch::Tensor idx, int64_t max_new_tokens,
                              int64_t context_length) {
  // @step eval
  model->eval();
  const torch::NoGradGuard no_grad;
  for (int64_t i = 0; i < max_new_tokens; ++i) {
    // @step crop
    const auto start = std::max<int64_t>(0, idx.size(1) - context_length);
    const auto logits = model(idx.index({Slice(), Slice(start, None)})).index({Slice(), -1});
    // @step pick
    const auto next_id = logits.argmax(/*dim=*/-1, /*keepdim=*/true);
    idx = torch::cat({idx, next_id}, /*dim=*/1);
  }
  return idx;
}

// @twin run_gpt
void run_gpt(const Preset& preset) {
  // @step build
  const auto& cfg = preset.model;
  const Tokenizer tokenizer;
  torch::manual_seed(preset.seed);
  GPTModel model(cfg);

  // @step logits
  report::section("GPTModel");
  report::count("parameters", count_parameters(*model));
  const auto prompt = torch::tensor(tokenizer.encode("Hello, I am")).unsqueeze(0);
  report::show("prompt ids", prompt);
  model->eval();
  {
    const torch::NoGradGuard no_grad;
    report::show("logits", model(prompt));
  }

  // @step generate
  report::section("generate_greedy");
  const auto out = generate_greedy(model, prompt, /*max_new_tokens=*/6, cfg.context_length);
  report::show("generated ids", out);
  const auto row = out[0].contiguous();
  report::text("decoded",
               tokenizer.decode({row.data_ptr<int64_t>(), row.data_ptr<int64_t>() + row.numel()}));
}

}  // namespace newtollm

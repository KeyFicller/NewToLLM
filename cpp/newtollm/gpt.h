#pragma once
// Lesson 05 - the GPT model (twin of py/newtollm/gpt.py).

#include "newtollm/attention.h"
#include "newtollm/config.h"
#include "newtollm/modules.h"

#include <torch/torch.h>

#include <cstdint>

namespace newtollm {

// @twin TransformerBlock
class TransformerBlockImpl : public torch::nn::Module {
 public:
  explicit TransformerBlockImpl(const GPTConfig& cfg);
  torch::Tensor forward(torch::Tensor x);

 private:
  MultiHeadAttention att_;
  FeedForward ff_;
  LayerNorm norm1_;
  LayerNorm norm2_;
  torch::nn::Dropout drop_shortcut_;
};
TORCH_MODULE(TransformerBlock);

// @twin GPTModel
class GPTModelImpl : public torch::nn::Module {
 public:
  explicit GPTModelImpl(const GPTConfig& cfg);
  torch::Tensor forward(const torch::Tensor& idx);
  // Swap the vocabulary projection for a fresh `num_outputs`-way head (lesson 07).
  void replace_head(int64_t num_outputs);

 private:
  torch::nn::Embedding tok_emb_;
  torch::nn::Embedding pos_emb_;
  torch::nn::Dropout drop_emb_;
  torch::nn::Sequential trf_blocks_;
  LayerNorm final_norm_;
  torch::nn::Linear out_head_;
};
TORCH_MODULE(GPTModel);
// @endtwin

[[nodiscard]] torch::Tensor generate_greedy(GPTModel& model, torch::Tensor idx,
                                            int64_t max_new_tokens, int64_t context_length);
void run_gpt(const Preset& preset);

}  // namespace newtollm

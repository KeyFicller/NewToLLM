#pragma once
// Lesson 03 - modules (twin of py/newtollm/modules.py).

#include "newtollm/config.h"

#include <torch/torch.h>

#include <cstdint>

namespace newtollm {

// @twin GELU
// tanh approximation used by GPT-2.
class GELUImpl : public torch::nn::Module {
 public:
  torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(GELU);

// @twin LayerNorm
class LayerNormImpl : public torch::nn::Module {
 public:
  explicit LayerNormImpl(int64_t emb_dim, double eps = 1e-5);
  torch::Tensor forward(const torch::Tensor& x);

 private:
  double eps_;
  torch::Tensor scale_;
  torch::Tensor shift_;
};
TORCH_MODULE(LayerNorm);

// @twin FeedForward
class FeedForwardImpl : public torch::nn::Module {
 public:
  explicit FeedForwardImpl(int64_t emb_dim);
  torch::Tensor forward(const torch::Tensor& x);

 private:
  torch::nn::Sequential layers_;
};
TORCH_MODULE(FeedForward);

[[nodiscard]] int64_t count_parameters(const torch::nn::Module& module);
void run_modules(const Preset& preset);

}  // namespace newtollm

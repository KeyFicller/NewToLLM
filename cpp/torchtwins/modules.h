#pragma once
// Lesson 03 - modules (twin of py/torchtwins/modules.py).

#include "torchtwins/config.h"

#include <torch/torch.h>

#include <cstdint>

namespace torchtwins {

// @twin GELU
// tanh approximation used by GPT-2.
class GELUImpl : public torch::nn::Module {
 public:
  torch::Tensor forward(const torch::Tensor& x);
};
// TORCH_MODULE(X) generates class X: a thin holder around std::shared_ptr<XImpl>.
// Copying X copies the pointer (shared ownership, like a Python reference),
// `x->method()` reaches XImpl's members, and `x(input)` forwards to XImpl::forward.
// That is why the class above is named ...Impl and call sites use `->`.
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
// @endtwin

[[nodiscard]] int64_t count_parameters(const torch::nn::Module& module);
void run_modules(const Preset& preset);

}  // namespace torchtwins

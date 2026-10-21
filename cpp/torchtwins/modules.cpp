// Lesson 03 - modules: GELU, LayerNorm and the Transformer feed-forward block.
#include "torchtwins/modules.h"

#include "torchtwins/report.h"

#include <numbers>

namespace torchtwins {

// @twin GELU
// @step formula
torch::Tensor GELUImpl::forward(const torch::Tensor& x) {
  return 0.5 * x *
         (1.0 + torch::tanh(std::sqrt(2.0 / std::numbers::pi) * (x + 0.044715 * x.pow(3))));
}

// @twin LayerNorm
// @step params
LayerNormImpl::LayerNormImpl(int64_t emb_dim, double eps)
    : eps_(eps),
      scale_(register_parameter("scale", torch::ones({emb_dim}))),
      shift_(register_parameter("shift", torch::zeros({emb_dim}))) {}

torch::Tensor LayerNormImpl::forward(const torch::Tensor& x) {
  // @step stats
  const auto mean = x.mean(/*dim=*/-1, /*keepdim=*/true);
  const auto var = x.var(/*dim=*/-1, /*unbiased=*/false, /*keepdim=*/true);
  // @step normalize
  return scale_ * (x - mean) / torch::sqrt(var + eps_) + shift_;
}

// @twin FeedForward
// @step layers
FeedForwardImpl::FeedForwardImpl(int64_t emb_dim)
    : layers_(register_module("layers", torch::nn::Sequential())) {
  // push_back one by one: C++ leaves the evaluation order of constructor
  // arguments unspecified (GCC goes right-to-left), which would draw the
  // second Linear's random init before the first one's.
  layers_->push_back(torch::nn::Linear(emb_dim, 4 * emb_dim));
  layers_->push_back(GELU());
  layers_->push_back(torch::nn::Linear(4 * emb_dim, emb_dim));
}

// @step forward
torch::Tensor FeedForwardImpl::forward(const torch::Tensor& x) {
  return layers_->forward<torch::Tensor>(x);
}

// @twin count_parameters
int64_t count_parameters(const torch::nn::Module& module) {
  // @step sum
  int64_t total = 0;
  for (const auto& p : module.parameters()) {
    total += p.numel();
  }
  return total;
}

// @twin run_modules
void run_modules(const Preset& preset) {
  // @step input
  const auto emb_dim = preset.model.emb_dim;
  torch::manual_seed(preset.seed);
  const auto x = torch::randn({2, 3, emb_dim});

  // @step gelu
  report::section("GELU");
  report::show("gelu(linspace(-3, 3, 7))", GELU()(torch::linspace(-3.0, 3.0, 7)));

  // @step layernorm
  report::section("LayerNorm");
  const auto normed = LayerNorm(emb_dim)(x);
  report::show("mean(-1)", normed.mean(/*dim=*/-1));
  report::show("var(-1)", normed.var(/*dim=*/-1, /*unbiased=*/false));

  // @step ff
  report::section("FeedForward");
  FeedForward ff(emb_dim);
  report::count("parameters", count_parameters(*ff));
  report::show("ff(x)", ff(x));
}

}  // namespace torchtwins

// Lesson 02 - autograd: gradients, no_grad, and a hand-written SGD loop.
#include "newtollm/autograd.h"

#include "newtollm/report.h"

#include <torch/torch.h>

namespace newtollm {

namespace {
std::string bool_text(bool value) { return value ? "true" : "false"; }
}  // namespace

// @twin gradients
void gradients() {
  report::section("gradients");
  const auto x = torch::tensor({1.0F, 2.0F, 3.0F});
  const auto target = torch::tensor({2.0F, 4.0F, 6.0F});
  auto w = torch::tensor(0.5F, torch::requires_grad());
  auto b = torch::tensor(0.1F, torch::requires_grad());
  auto loss = (w * x + b - target).pow(2).mean();
  loss.backward();
  report::scalar("loss", loss.item<double>());
  report::show("w.grad", w.grad());
  report::show("b.grad", b.grad());
}

// @twin no_grad
void no_grad() {
  report::section("no_grad");
  const auto w = torch::ones({3}, torch::requires_grad());
  torch::Tensor frozen;
  {
    const torch::NoGradGuard guard;
    frozen = w * 2;
  }
  const auto tracked = w * 2;
  report::text("frozen.requires_grad", bool_text(frozen.requires_grad()));
  report::text("tracked.requires_grad", bool_text(tracked.requires_grad()));
}

// @twin fit_line
void fit_line(uint64_t seed, int64_t steps, double lr) {
  report::section("fit_line");
  torch::manual_seed(seed);
  const auto x = torch::randn({32, 1});
  const auto y = 3.0 * x + 0.5 + 0.1 * torch::randn({32, 1});
  auto w = torch::zeros({1}, torch::requires_grad());
  auto b = torch::zeros({1}, torch::requires_grad());
  auto loss = torch::tensor(0.0F);
  for (int64_t step = 0; step < steps; ++step) {
    loss = (x * w + b - y).pow(2).mean();
    loss.backward();
    {
      const torch::NoGradGuard guard;
      w -= lr * w.grad();
      b -= lr * b.grad();
      w.mutable_grad().zero_();
      b.mutable_grad().zero_();
    }
  }
  report::show("w", w);
  report::show("b", b);
  report::scalar("final loss", loss.item<double>());
}

// @twin run_autograd
void run_autograd(const Preset& preset) {
  gradients();
  no_grad();
  fit_line(preset.seed);
}

}  // namespace newtollm

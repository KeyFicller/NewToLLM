// Lesson 02 - autograd: gradients, no_grad, and a hand-written SGD loop.
#include "torchtwins/autograd.h"

#include "torchtwins/report.h"

#include <torch/torch.h>

namespace torchtwins {

namespace {
std::string bool_text(bool value) { return value ? "true" : "false"; }
}  // namespace

// @twin gradients
void gradients() {
  report::section("gradients");
  // @step data
  const auto x = torch::tensor({1.0F, 2.0F, 3.0F});
  const auto target = torch::tensor({2.0F, 4.0F, 6.0F});
  // @step leaf
  auto w = torch::tensor(0.5F, torch::requires_grad());
  auto b = torch::tensor(0.1F, torch::requires_grad());
  // @step loss
  auto loss = (w * x + b - target).pow(2).mean();
  // @step backward
  loss.backward();
  // @step read
  report::scalar("loss", loss.item<double>());
  report::show("w.grad", w.grad());
  report::show("b.grad", b.grad());
}

// @twin no_grad
void no_grad() {
  report::section("no_grad");
  // @step leaf
  const auto w = torch::ones({3}, torch::requires_grad());
  // @step guard
  torch::Tensor frozen;
  {
    const torch::NoGradGuard guard;
    frozen = w * 2;
  }
  // @step tracked
  const auto tracked = w * 2;
  // @step report
  report::text("frozen.requires_grad", bool_text(frozen.requires_grad()));
  report::text("tracked.requires_grad", bool_text(tracked.requires_grad()));
}

// @twin fit_line
void fit_line(uint64_t seed, int64_t steps, double lr) {
  report::section("fit_line");
  // @step data
  torch::manual_seed(seed);
  const auto x = torch::randn({32, 1});
  const auto y = 3.0 * x + 0.5 + 0.1 * torch::randn({32, 1});
  // @step params
  auto w = torch::zeros({1}, torch::requires_grad());
  auto b = torch::zeros({1}, torch::requires_grad());
  auto loss = torch::tensor(0.0F);
  for (int64_t step = 0; step < steps; ++step) {
    // @step forward
    loss = (x * w + b - y).pow(2).mean();
    loss.backward();
    {
      // @step sgd
      const torch::NoGradGuard guard;
      w -= lr * w.grad();
      b -= lr * b.grad();
      w.mutable_grad().zero_();
      b.mutable_grad().zero_();
    }
  }
  // @step report
  report::show("w", w);
  report::show("b", b);
  report::scalar("final loss", loss.item<double>());
}

// @twin run_autograd
void run_autograd(const Preset& preset) {
  // @step pipeline
  gradients();
  no_grad();
  fit_line(preset.seed);
}

}  // namespace torchtwins

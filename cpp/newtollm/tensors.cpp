// Lesson 01 - tensors: creation, shapes, indexing, broadcasting, matmul.
#include "newtollm/tensors.h"

#include "newtollm/report.h"

namespace newtollm {

using torch::indexing::None;
using torch::indexing::Slice;

// @twin create
torch::Tensor create(uint64_t seed) {
  report::section("create");
  auto x = torch::arange(6, torch::kFloat32).reshape({2, 3});
  report::show("arange.reshape(2, 3)", x);
  report::show("zeros(2, 2)", torch::zeros({2, 2}));
  report::show("full((2, 2), 7)", torch::full({2, 2}, 7.0));
  torch::manual_seed(seed);
  report::show("randn(2, 3)", torch::randn({2, 3}));
  report::show("tensor([[1, 2], [3, 4]])", torch::tensor({1, 2, 3, 4}, torch::kLong).view({2, 2}));
  return x;
}

// @twin reshape
void reshape(const torch::Tensor& x) {
  report::section("reshape");
  report::show("x.view(3, 2)", x.view({3, 2}));
  report::show("x.transpose(0, 1)", x.transpose(0, 1));
  report::show("x.unsqueeze(0)", x.unsqueeze(0));
  report::show("x.flatten()", x.flatten());
  report::count("x.size(-1)", x.size(-1));
}

// @twin index
void index(const torch::Tensor& x) {
  report::section("index");
  report::show("x[0]", x[0]);
  report::show("x[:, 1:]", x.index({Slice(), Slice(1, None)}));
  report::show("x[:, -1]", x.index({Slice(), -1}));
  report::show("x > 2", x > 2);
  report::show("x[x > 2]", x.index({x > 2}));
}

// @twin broadcast
void broadcast(const torch::Tensor& x) {
  report::section("broadcast");
  const auto bias = torch::tensor({10.0F, 20.0F, 30.0F});
  report::show("x + bias", x + bias);
  report::show("x * 2", x * 2);
  report::show("x.mean(dim=-1, keepdim=True)", x.mean(/*dim=*/-1, /*keepdim=*/true));
}

// @twin matmul
void matmul(const torch::Tensor& x) {
  report::section("matmul");
  report::show("x @ x.T", torch::matmul(x, x.t()));
  report::show("softmax(x, dim=-1)", torch::softmax(x, /*dim=*/-1));
}

// @twin run_tensors
void run_tensors(const Preset& preset) {
  const auto x = create(preset.seed);
  reshape(x);
  index(x);
  broadcast(x);
  matmul(x);
}

}  // namespace newtollm

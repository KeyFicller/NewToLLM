// Lesson 01 - tensors: creation, shapes, indexing, broadcasting, matmul.
#include "torchtwins/tensors.h"

#include "torchtwins/report.h"

namespace torchtwins {

using torch::indexing::None;
using torch::indexing::Slice;

// @twin create
torch::Tensor create(uint64_t seed) {
  report::section("create");
  // @step make
  // Shape (2, 3): 2 rows x 3 cols. reshape() returns a *view* when memory allows -
  // no data is copied, only the stride metadata changes. Shapes are brace lists here.
  auto x = torch::arange(6, torch::kFloat32).reshape({2, 3});
  report::show("arange.reshape(2, 3)", x);
  report::show("zeros(2, 2)", torch::zeros({2, 2}));
  report::show("full((2, 2), 7)", torch::full({2, 2}, 7.0));
  // @step seeded
  torch::manual_seed(seed);
  report::show("randn(2, 3)", torch::randn({2, 3}));
  // @step literal
  report::show("tensor([[1, 2], [3, 4]])", torch::tensor({1, 2, 3, 4}, torch::kLong).view({2, 2}));
  return x;
}

// @twin reshape
void reshape(const torch::Tensor& x) {
  report::section("reshape");
  // @step view
  // view() needs contiguous memory; transpose() only swaps strides, so a transposed
  // tensor must be .contiguous()'d before another view() (see lesson 04, merge step).
  report::show("x.view(3, 2)", x.view({3, 2}));
  report::show("x.transpose(0, 1)", x.transpose(0, 1));
  report::show("x.unsqueeze(0)", x.unsqueeze(0));
  report::show("x.flatten()", x.flatten());
  // @step size
  report::count("x.size(-1)", x.size(-1));
}

// @twin index
void index(const torch::Tensor& x) {
  report::section("index");
  // @step int
  report::show("x[0]", x[0]);
  // @step slice
  // Python's x[:, 1:] is spelled index({Slice(), Slice(1, None)}) in LibTorch.
  report::show("x[:, 1:]", x.index({Slice(), Slice(1, None)}));
  report::show("x[:, -1]", x.index({Slice(), -1}));
  // @step mask
  report::show("x > 2", x > 2);
  report::show("x[x > 2]", x.index({x > 2}));
}

// @twin broadcast
void broadcast(const torch::Tensor& x) {
  report::section("broadcast");
  // @step bias
  const auto bias = torch::tensor({10.0F, 20.0F, 30.0F});
  // @step ops
  // Broadcasting aligns shapes from the right: (2, 3) + (3,) -> (2, 3).
  report::show("x + bias", x + bias);
  report::show("x * 2", x * 2);
  // @step reduce
  report::show("x.mean(dim=-1, keepdim=True)", x.mean(/*dim=*/-1, /*keepdim=*/true));
}

// @twin matmul
void matmul(const torch::Tensor& x) {
  report::section("matmul");
  // @step matmul
  // (2, 3) @ (3, 2) -> (2, 2). The inner dims must agree.
  // C++ has no @ operator for tensors; torch::matmul is the same op.
  report::show("x @ x.T", torch::matmul(x, x.t()));
  // @step softmax
  report::show("softmax(x, dim=-1)", torch::softmax(x, /*dim=*/-1));
}

// @twin run_tensors
void run_tensors(const Preset& preset) {
  // @step pipeline
  const auto x = create(preset.seed);
  reshape(x);
  index(x);
  broadcast(x);
  matmul(x);
}

}  // namespace torchtwins

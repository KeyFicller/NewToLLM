// Lesson 08 - checkpoints: save weights, reload into a fresh model.
#include "torchtwins/checkpoint.h"

#include "torchtwins/modules.h"
#include "torchtwins/report.h"
#include "torchtwins/tokenizer.h"

#include <torch/serialize.h>

#include <unistd.h>
#include <cstdint>
#include <filesystem>
#include <format>

namespace torchtwins {

// @twin save_weights
void save_weights(GPTModel& model, const std::filesystem::path& path) {
  // @step dump
  // torch::save walks the Module via OutputArchive (parameters + buffers).
  // That is *not* Python's zip+pickle state_dict — do not cross-load the files.
  torch::save(model, path.string());
}

// @twin load_weights
void load_weights(GPTModel& model, const std::filesystem::path& path) {
  // @step read
  // InputArchive fills tensors already registered on this Module by name.
  // @step apply
  // Architecture must match the saved module (same GPTConfig).
  torch::load(model, path.string());
}

// @twin tensor_count
int64_t tensor_count(GPTModel& model) {
  // @step count
  // Parameters only. LibTorch has no persistent=False: register_buffer always
  // saves, so buffers().size() would be +2 vs Python's state_dict (the causal
  // masks). Masks are rebuilt from context_length anyway.
  return static_cast<int64_t>(model->parameters().size());
}

// @twin max_abs_diff
double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  // @step diff
  return (a - b).abs().max().item<double>();
}

// @twin run_checkpoint
void run_checkpoint(const Preset& preset) {
  const auto& cfg = preset.model;
  const Tokenizer tokenizer;

  // @step build
  torch::manual_seed(preset.seed);
  GPTModel model(cfg);
  model->eval();
  const auto prompt = torch::tensor(tokenizer.encode("Hello, I am")).unsqueeze(0);

  report::section("build");
  report::count("parameters", count_parameters(*model));
  report::count("parameter tensors", tensor_count(model));
  report::show("prompt ids", prompt);
  torch::Tensor fresh;
  {
    const torch::NoGradGuard no_grad;
    fresh = model(prompt);
  }
  report::show("fresh logits", fresh);

  // @step save
  report::section("save");
  const auto path =
      std::filesystem::temp_directory_path() / std::format("torchtwins_ckpt_{}.pt", ::getpid());
  save_weights(model, path);
  report::text("status", "saved");

  // @step scramble
  report::section("scramble");
  torch::manual_seed(preset.seed + 1);
  GPTModel other(cfg);
  other->eval();
  torch::Tensor scrambled;
  {
    const torch::NoGradGuard no_grad;
    scrambled = other(prompt);
  }
  report::show("scrambled logits", scrambled);

  // @step load
  report::section("load");
  load_weights(other, path);
  torch::Tensor restored;
  {
    const torch::NoGradGuard no_grad;
    restored = other(prompt);
  }
  report::show("restored logits", restored);
  // After a correct round-trip this is ~0; proves the file carried the weights.
  report::scalar("max abs diff", max_abs_diff(fresh, restored));
  std::filesystem::remove(path);
}

}  // namespace torchtwins

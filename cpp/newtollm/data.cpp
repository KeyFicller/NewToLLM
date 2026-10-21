// Lesson 06a - data pipeline: tokens -> train/val split -> sliding windows -> batches.
#include "newtollm/data.h"

#include "newtollm/report.h"

#include <ATen/CPUGeneratorImpl.h>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace newtollm {

using torch::indexing::None;
using torch::indexing::Slice;

std::filesystem::path default_text_path() { return repo_root() / "assets" / "the-verdict.txt"; }

std::string read_text(const std::filesystem::path& path) {
  const std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  std::stringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

namespace {
torch::Tensor encode_tensor(const Tokenizer& tokenizer, const std::filesystem::path& path) {
  return torch::tensor(tokenizer.encode(read_text(path)), torch::kLong);
}
}  // namespace

// @twin split_ids
std::pair<torch::Tensor, torch::Tensor> split_ids(const torch::Tensor& ids, double train_ratio) {
  // Split on *tokens*, not characters: str length (code points) and std::string
  // length (bytes) disagree on non-ASCII text, token counts never do.
  const auto cut = static_cast<int64_t>(train_ratio * static_cast<double>(ids.numel()));
  return {ids.index({Slice(None, cut)}), ids.index({Slice(cut, None)})};
}

// @twin make_windows
Windows make_windows(const torch::Tensor& ids, int64_t context_length, int64_t stride) {
  const auto inputs = ids.index({Slice(None, -1)}).unfold(0, context_length, stride);
  const auto targets = ids.index({Slice(1, None)}).unfold(0, context_length, stride);
  return {.inputs = inputs.contiguous(), .targets = targets.contiguous()};
}

// @twin batches
std::vector<Batch> batches(const Windows& windows, int64_t batch_size,
                           std::optional<at::Generator> generator) {
  // Full batches only (drop_last); shuffle with randperm when given a generator.
  const auto n = windows.size();
  const auto order = generator ? torch::randperm(n, *generator, torch::kLong) : torch::arange(n);
  std::vector<Batch> out;
  for (int64_t start = 0; start + batch_size <= n; start += batch_size) {
    const auto idx = order.index({Slice(start, start + batch_size)});
    out.emplace_back(windows.inputs.index({idx}), windows.targets.index({idx}));
  }
  return out;
}

// @twin load_dataset
Dataset load_dataset(const Preset& preset, const Tokenizer& tokenizer,
                     const std::filesystem::path& path) {
  const auto ids = encode_tensor(tokenizer, path);
  const auto [train_ids, val_ids] = split_ids(ids, preset.train.train_ratio);
  const auto ctx = preset.model.context_length;
  const auto stride = preset.train.stride;
  return {.train = make_windows(train_ids, ctx, stride), .val = make_windows(val_ids, ctx, stride)};
}

// @twin run_data
void run_data(const Preset& preset) {
  const Tokenizer tokenizer;
  report::section("tokens");
  const auto ids = encode_tensor(tokenizer, default_text_path());
  report::show("ids", ids);
  const auto [train_ids, val_ids] = split_ids(ids, preset.train.train_ratio);
  report::count("train tokens", train_ids.numel());
  report::count("val tokens", val_ids.numel());

  report::section("windows");
  const auto data = load_dataset(preset, tokenizer);
  report::show("train inputs", data.train.inputs);
  report::show("train targets", data.train.targets);
  report::count("train windows", data.train.size());
  report::count("val windows", data.val.size());

  report::section("batches");
  const auto generator = at::make_generator<at::CPUGeneratorImpl>(preset.seed);
  const auto train_batches = batches(data.train, preset.train.batch_size, generator);
  report::count("train batches per epoch", static_cast<int64_t>(train_batches.size()));
  report::count("val batches",
                static_cast<int64_t>(batches(data.val, preset.train.batch_size).size()));
  report::show("first batch inputs", train_batches.front().first);
}

}  // namespace newtollm

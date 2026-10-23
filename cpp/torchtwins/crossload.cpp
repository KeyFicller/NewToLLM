// Lesson 09 - portable weight bridge across Python and C++.
#include "torchtwins/crossload.h"

#include "torchtwins/modules.h"
#include "torchtwins/report.h"
#include "torchtwins/tokenizer.h"

#include <torch/serialize.h>

#include <unistd.h>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torchtwins {
namespace {

std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::vector<char>& bytes) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("cannot write " + path.string());
  }
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

}  // namespace

// @twin export_portable
void export_portable(GPTModel& model, const std::filesystem::path& path) {
  // @step collect
  // named_parameters() order matches Python (parity gate). Names ride along so
  // import can match by key — the same skill an OpenAI/HF remap will need.
  // NOLINTNEXTLINE(misc-const-correctness) — List::push_back needs a mutable handle
  c10::List<std::string> names;
  // NOLINTNEXTLINE(misc-const-correctness)
  c10::List<at::Tensor> tensors;
  for (const auto& item : model->named_parameters()) {
    names.push_back(item.key());
    tensors.push_back(item.value().detach().contiguous());
  }
  // @step dump
  // pickle_save writes the IValue protocol Python's torch.load already reads.
  // This is *not* torch::save(Module) (OutputArchive) and *not* a Python
  // state_dict zip of nn.Module keys+buffers — only the portable pair.
  const auto bytes = torch::pickle_save(c10::ivalue::Tuple::create({names, tensors}));
  write_bytes(path, bytes);
}

// @twin import_portable
void import_portable(GPTModel& model, const std::filesystem::path& path) {
  // @step read
  // pickle_load accepts both LibTorch pickle_save bytes and Python torch.save
  // zip containers of the same (names, tensors) pair.
  const auto root = torch::pickle_load(read_bytes(path)).toTuple();
  const auto name_list = root->elements().at(0).toList();
  const auto tensor_list = root->elements().at(1).toList();
  if (name_list.size() != tensor_list.size()) {
    throw std::runtime_error("portable checkpoint name/tensor count mismatch");
  }
  // @step apply
  std::unordered_map<std::string, torch::Tensor> table;
  table.reserve(name_list.size());
  for (size_t i = 0; i < name_list.size(); ++i) {
    table.emplace(std::string(name_list.get(i).toStringRef()), tensor_list.get(i).toTensor());
  }
  std::unordered_set<std::string> expected;
  for (const auto& item : model->named_parameters()) {
    expected.insert(item.key());
  }
  if (table.size() != expected.size()) {
    throw std::runtime_error("portable checkpoint key count mismatch");
  }
  for (const auto& key : expected) {
    if (!table.contains(key)) {
      throw std::runtime_error("portable checkpoint missing key: " + key);
    }
  }
  {
    const torch::NoGradGuard no_grad;
    for (auto& item : model->named_parameters()) {
      item.value().copy_(table.at(item.key()));
    }
  }
}

// @twin param_names
std::vector<std::string> param_names(GPTModel& model) {
  // @step list
  std::vector<std::string> names;
  for (const auto& item : model->named_parameters()) {
    names.push_back(item.key());
  }
  return names;
}

// @twin logits_diff
double logits_diff(const torch::Tensor& a, const torch::Tensor& b) {
  // @step diff
  return (a - b).abs().max().item<double>();
}

// @twin run_crossload
void run_crossload(const Preset& preset) {
  const auto& cfg = preset.model;
  const Tokenizer tokenizer;

  // @step build
  torch::manual_seed(preset.seed);
  GPTModel model(cfg);
  model->eval();
  const auto prompt = torch::tensor(tokenizer.encode("Hello, I am")).unsqueeze(0);
  const auto names = param_names(model);

  report::section("build");
  report::count("parameters", count_parameters(*model));
  report::count("parameter tensors", static_cast<int64_t>(names.size()));
  report::text("first parameter", names.front());
  report::show("prompt ids", prompt);
  torch::Tensor fresh;
  {
    const torch::NoGradGuard no_grad;
    fresh = model(prompt);
  }
  report::show("fresh logits", fresh);

  // @step export
  report::section("export");
  const auto path =
      std::filesystem::temp_directory_path() / std::format("torchtwins_portable_{}.pt", ::getpid());
  export_portable(model, path);
  report::text("format", "portable pickle (names, tensors)");
  report::text("status", "exported");

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

  // @step import
  report::section("import");
  import_portable(other, path);
  torch::Tensor restored;
  {
    const torch::NoGradGuard no_grad;
    restored = other(prompt);
  }
  report::show("restored logits", restored);
  report::scalar("max abs diff", logits_diff(fresh, restored));
  std::filesystem::remove(path);
}

}  // namespace torchtwins

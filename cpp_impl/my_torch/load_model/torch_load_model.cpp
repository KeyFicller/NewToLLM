#include "my_torch.h"
#include "toy_model/model.h"

#include <filesystem>
#include <iostream>
#include <pytorch/tokenizers/tiktoken.h>
#include <stdexcept>
#include <torch/script.h>
#include <torch/torch.h>

namespace MyTorch {
namespace {

struct GenerationOptions {
  std::string prompt = "I love you , but you ";
  int64_t max_new_tokens = 30;
  int64_t context_length = 1024;
  int64_t eos_id = 50256;
};

constexpr const char *kTokenizerPath = "assets/gpt2.tiktoken";

std::filesystem::path resolve_scripted_model_path() {
  const std::vector<std::filesystem::path> candidates = {
      "python_impl/.temp/toy_model_from_gpt2.pt",
      "cpp_impl/.temp/toy_model_from_gpt2.pt"};
  for (const auto &p : candidates) {
    if (std::filesystem::exists(p)) {
      return p;
    }
  }
  return {};
}

std::filesystem::path resolve_state_dict_path() {
  const std::vector<std::filesystem::path> candidates = {
      "python_impl/.temp/toy_model_from_gpt2.pth",
      "cpp_impl/.temp/toy_model_from_gpt2.pth"};
  for (const auto &p : candidates) {
    if (std::filesystem::exists(p)) {
      return p;
    }
  }
  return {};
}

std::vector<uint64_t> encode_prompt(const std::string &prompt) {
  auto *tokenizer = get_tokenizer(kTokenizerPath);
  if (tokenizer == nullptr) {
    return {};
  }
  auto encoded = tokenizer->encode(prompt, 0, 0);
  if (!encoded.ok()) {
    return {};
  }
  return *encoded;
}

std::string decode_ids(const std::vector<uint64_t> &ids) {
  auto *tokenizer = get_tokenizer(kTokenizerPath);
  if (tokenizer == nullptr) {
    return "<decode skipped: tokenizer unavailable>";
  }

  std::string output;
  uint64_t prev = 0;
  bool has_prev = false;
  for (const auto token : ids) {
    auto piece = tokenizer->decode(has_prev ? prev : token, token, true);
    if (piece.ok()) {
      output += *piece;
    }
    prev = token;
    has_prev = true;
  }
  return output;
}

std::vector<uint64_t> tensor_to_ids(const torch::Tensor &out) {
  auto out_flat = out.squeeze(0).to(torch::kLong).to(torch::kCPU);
  std::vector<uint64_t> out_ids;
  out_ids.reserve(static_cast<size_t>(out_flat.size(0)));
  for (int64_t i = 0; i < out_flat.size(0); ++i) {
    out_ids.push_back(static_cast<uint64_t>(out_flat[i].item<int64_t>()));
  }
  return out_ids;
}

void print_generation_output(const std::string &loaded_from,
                             const torch::Tensor &out) {
  std::cout << loaded_from << '\n';
  std::cout << "[load_model] Toy model output: "
            << decode_ids(tensor_to_ids(out)) << '\n';
}

torch::Tensor
generate_text_scripted(torch::jit::script::Module &model, torch::Tensor idx,
                       int64_t max_new_tokens, int64_t context_length,
                       c10::optional<int64_t> eos_id = c10::nullopt) {
  using torch::indexing::None;
  using torch::indexing::Slice;

  for (int64_t i = 0; i < max_new_tokens; ++i) {
    const auto start = std::max<int64_t>(0, idx.size(1) - context_length);
    auto idx_cond = idx.index({Slice(), Slice(start, None)});
    auto logits = model.forward({idx_cond}).toTensor();
    logits = logits.index({Slice(), -1, Slice()});
    auto idx_next = std::get<1>(logits.max(-1, true));

    if (eos_id.has_value() && idx_next.eq(*eos_id).all().item<bool>()) {
      break;
    }
    idx = torch::cat({idx, idx_next}, 1);
  }
  return idx;
}

} // namespace

torch::jit::script::Module load_saved_toy_model_jit() {
  const auto scripted_path = resolve_scripted_model_path();
  if (scripted_path.empty()) {
    throw std::runtime_error(
        "[load_model] TorchScript file not found in cpp_impl/.temp or "
        "python_impl/.temp");
  }

  try {
    auto scripted_model = torch::jit::load(scripted_path.string());
    std::cout << "[load_model] Loaded TorchScript from: "
              << scripted_path.string() << '\n';
    return scripted_model;
  } catch (const c10::Error &e) {
    throw std::runtime_error("[load_model] failed to load TorchScript: " +
                             std::string(e.msg()));
  }
}

void load_model_torch() {
  const GenerationOptions gen_opts;
  auto encoded_ids = encode_prompt(gen_opts.prompt);
  if (encoded_ids.empty()) {
    std::cout << "[load_model] failed to encode prompt with gpt2 tokenizer\n";
    return;
  }

  std::vector<int64_t> encoded_i64(encoded_ids.begin(), encoded_ids.end());
  auto idx = torch::tensor(encoded_i64, torch::kLong).unsqueeze(0);

  // Note: Use model saved by python implementation.

  // Preferred path: load TorchScript module exported by Python.
  const auto scripted_path = resolve_scripted_model_path();
  if (!scripted_path.empty()) {
    try {
      auto scripted_model = load_saved_toy_model_jit();
      scripted_model.eval();
      torch::NoGradGuard no_grad;
      auto out = generate_text_scripted(
          scripted_model, idx.clone(), gen_opts.max_new_tokens,
          gen_opts.context_length, gen_opts.eos_id);

      print_generation_output("[load_model] Loaded TorchScript from: " +
                                  scripted_path.string(),
                              out);
      return;
    } catch (const std::exception &e) {
      std::cout << "[load_model] failed to load TorchScript from "
                << scripted_path << '\n';
      std::cout << e.what() << '\n';
    }
  }
}

} // namespace MyTorch

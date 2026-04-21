#include "my_torch.h"
#include "toy_model/model.h"

#include <filesystem>
#include <iostream>
#include <pytorch/tokenizers/tiktoken.h>
#include <torch/torch.h>

namespace MyTorch {
namespace {

std::vector<uint64_t> encode_prompt_with_tiktoken(const std::string &prompt,
                                                  bool &used_tiktoken) {
  used_tiktoken = false;

  const std::vector<std::filesystem::path> candidates = {
      "assets/gpt2.tiktoken"};

  std::filesystem::path tokenizer_path;
  for (const auto &path : candidates) {
    if (std::filesystem::exists(path)) {
      tokenizer_path = path;
      break;
    }
  }

  if (tokenizer_path.empty()) {
    std::cout << "[toy_model] tokenizer file not found in assets/, fallback to "
                 "hardcoded token ids.\n";
    return {15496, 11, 314, 716};
  }

  tokenizers::Tiktoken tokenizer;
  const auto load_error = tokenizer.load(tokenizer_path.string());
  if (load_error != tokenizers::Error::Ok) {
    std::cout << "[toy_model] failed to load tiktoken file: " << tokenizer_path
              << ", fallback to hardcoded token ids.\n";
    return {15496, 11, 314, 716};
  }

  auto encode_result = tokenizer.encode(prompt, 0, 0);
  if (!encode_result.ok()) {
    std::cout << "[toy_model] failed to encode prompt with tiktoken, fallback "
                 "to hardcoded token ids.\n";
    return {15496, 11, 314, 716};
  }

  used_tiktoken = true;
  return std::move(*encode_result);
}

std::string decode_tokens_with_tiktoken(const std::vector<uint64_t> &token_ids,
                                        bool tokenizer_was_used) {
  if (!tokenizer_was_used) {
    return "<decode skipped: tiktoken artifact not available>";
  }

  const std::vector<std::filesystem::path> candidates = {
      "assets/gpt2.tiktoken"};

  std::filesystem::path tokenizer_path;
  for (const auto &path : candidates) {
    if (std::filesystem::exists(path)) {
      tokenizer_path = path;
      break;
    }
  }
  if (tokenizer_path.empty()) {
    return "<decode skipped: tokenizer file missing>";
  }

  tokenizers::Tiktoken tokenizer;
  if (tokenizer.load(tokenizer_path.string()) != tokenizers::Error::Ok) {
    return "<decode skipped: tokenizer load failed>";
  }

  std::string output;
  uint64_t prev = 0;
  bool has_prev = false;
  for (auto token : token_ids) {
    auto decode_result = tokenizer.decode(has_prev ? prev : token, token, true);
    if (!decode_result.ok()) {
      continue;
    }
    output += *decode_result;
    prev = token;
    has_prev = true;
  }
  return output;
}

} // namespace

void toy_model_torch() {
  torch::manual_seed(123);

  // Like Python config dict, but as a strongly typed C++ struct.
  ToyModelConfig cfg;
  ToyModel model(cfg);

  int64_t total_params = 0;
  for (const auto &p : model->parameters()) {
    total_params += p.numel();
  }
  std::cout << "[toy_model] Total parameters: " << total_params << '\n';

  bool used_tiktoken = false;
  const std::string prompt = "Hello, I am";
  auto encoded_ids = encode_prompt_with_tiktoken(prompt, used_tiktoken);
  auto encoded = torch::tensor(std::vector<int64_t>(encoded_ids.begin(),
                                                    encoded_ids.end()),
                               torch::kLong)
                     .unsqueeze(0);
  std::cout << "[toy_model] Encoded tensor shape: " << encoded.sizes() << '\n';

  // Inference mode in C++ typically combines eval() + NoGradGuard.
  model->eval();
  torch::NoGradGuard no_grad;
  auto output_ids = generate_text_simple(model, encoded, 6, cfg.context_length);
  std::cout << "[toy_model] Generated token ids: " << output_ids << '\n';
  std::cout << "[toy_model] Generated sequence length: " << output_ids.size(1)
            << '\n';

  std::vector<uint64_t> generated_ids;
  generated_ids.reserve(static_cast<size_t>(output_ids.size(1)));
  auto output_flat = output_ids.squeeze(0).to(torch::kLong);
  for (int64_t i = 0; i < output_flat.size(0); ++i) {
    generated_ids.push_back(
        static_cast<uint64_t>(output_flat[i].item<int64_t>()));
  }
  std::cout << "[toy_model] Decoded text: "
            << decode_tokens_with_tiktoken(generated_ids, used_tiktoken)
            << '\n';
}

} // namespace MyTorch

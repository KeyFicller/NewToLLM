#pragma once
// Lesson 07 - fine-tuning (twin of py/newtollm/finetune.py).

#include "newtollm/config.h"
#include "newtollm/data.h"
#include "newtollm/gpt.h"
#include "newtollm/tokenizer.h"

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace newtollm {

inline constexpr std::array<std::string_view, 2> kLabels = {"ham", "spam"};
using Example = std::pair<int64_t, std::string>;

[[nodiscard]] std::filesystem::path default_sms_path();

// @twin FinetuneHistory
struct FinetuneHistory {
  std::vector<double> train_loss;
  std::vector<double> train_accuracy;
  std::vector<double> val_accuracy;
};

[[nodiscard]] std::vector<Example> load_sms(
    const std::filesystem::path& path = default_sms_path());
[[nodiscard]] std::vector<Example> balanced(const std::vector<Example>& rows, int64_t per_class);
[[nodiscard]] Windows encode_examples(const std::vector<Example>& examples,
                                      const Tokenizer& tokenizer, int64_t context_length);
[[nodiscard]] std::tuple<Windows, Windows, Windows> split_examples(const Windows& windows);
void freeze_body(GPTModel& model, int64_t n_layers);
[[nodiscard]] torch::Tensor class_logits(GPTModel& model, const torch::Tensor& inputs);
[[nodiscard]] double accuracy(GPTModel& model, const Windows& windows, int64_t batch_size);
FinetuneHistory finetune_model(GPTModel& model, const Preset& preset, const Windows& train,
                               const Windows& val);
[[nodiscard]] std::string classify(GPTModel& model, const Tokenizer& tokenizer,
                                   const std::string& text, int64_t context_length);
void run_finetune(const Preset& preset);

}  // namespace newtollm

#pragma once
// Lesson 06b - pre-training (twin of py/newtollm/train.py).

#include "newtollm/config.h"
#include "newtollm/data.h"
#include "newtollm/gpt.h"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace newtollm {

// @twin TrainHistory
struct TrainHistory {
  // @step fields
  std::vector<int64_t> steps;
  std::vector<double> train_loss;
  std::vector<int64_t> eval_steps;
  std::vector<double> val_loss;
  int64_t tokens_seen = 0;
  double seconds = 0.0;
};

// @endtwin
[[nodiscard]] torch::Tensor loss_batch(GPTModel& model, const torch::Tensor& inputs,
                                       const torch::Tensor& targets);
[[nodiscard]] double evaluate(GPTModel& model, const Windows& windows, int64_t batch_size);
TrainHistory train_model(GPTModel& model, const Preset& preset, const Windows& train,
                         const Windows& val);
void save_history(const TrainHistory& history, const std::filesystem::path& path);
void run_train(const Preset& preset,
               const std::optional<std::filesystem::path>& curve_path = std::nullopt);

}  // namespace newtollm

#pragma once
// Lesson 06b - pre-training (twin of py/torchtwins/train.py).

#include "torchtwins/config.h"
#include "torchtwins/data.h"
#include "torchtwins/gpt.h"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace torchtwins {

// @twin TrainHistory
struct TrainHistory {
  // @step fields
  // steps / train_loss: one entry per optimizer step.
  // eval_steps / val_loss: sampled every eval_every steps.
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

}  // namespace torchtwins

#pragma once

// Runtime options + shared config (configs/toy_gpt2.json).
// Mirrors python_impl/utils/runtime.py. Set from sandbox flags:
// --seed, --device, --small.

#include "my_torch_export.h"
#include "toy_model/config.h"

#include <cstdint>
#include <optional>
#include <string>

namespace MyTorch {

struct RuntimeOptions {
  std::optional<uint64_t> seed;  // empty = use config
  std::string device;            // empty = auto-detect
  bool small = false;
};

struct TrainConfig {
  int64_t context_length = 256;
  int64_t batch_size = 2;
  double learning_rate = 4e-4;
  double weight_decay = 0.1;
  int64_t num_epochs = 10;
};

MY_TORCH_API RuntimeOptions &runtime_options();
MY_TORCH_API uint64_t runtime_seed();
// allow_small = false for tasks that load GPT-2 weights.
MY_TORCH_API ToyModelConfig model_config(bool allow_small = true);
MY_TORCH_API TrainConfig train_config();

} // namespace MyTorch

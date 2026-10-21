#pragma once
// Typed view of configs/presets.json (twin of py/newtollm/config.py).

#include <cstdint>
#include <filesystem>
#include <string>

namespace newtollm {

[[nodiscard]] std::filesystem::path repo_root();
[[nodiscard]] std::filesystem::path default_config_path();

// @twin GPTConfig
struct GPTConfig {
  int64_t vocab_size = 0;
  int64_t context_length = 0;
  int64_t emb_dim = 0;
  int64_t n_heads = 0;
  int64_t n_layers = 0;
  double drop_rate = 0.0;
  bool qkv_bias = false;
};

// @twin TrainConfig
struct TrainConfig {
  int64_t batch_size = 0;
  int64_t stride = 0;
  double learning_rate = 0.0;
  double weight_decay = 0.0;
  int64_t num_epochs = 0;
  int64_t eval_every = 0;
  double train_ratio = 0.0;
};

// @twin FinetuneConfig
struct FinetuneConfig {
  int64_t examples_per_class = 0;
  int64_t batch_size = 0;
  double learning_rate = 0.0;
  double weight_decay = 0.0;
  int64_t num_epochs = 0;
};

// @twin Preset
struct Preset {
  std::string name;
  uint64_t seed = 0;
  GPTConfig model;
  TrainConfig train;
  FinetuneConfig finetune;

  [[nodiscard]] Preset with_seed(uint64_t new_seed) const {
    Preset copy = *this;
    copy.seed = new_seed;
    return copy;
  }
};

[[nodiscard]] Preset load_preset(const std::string& name,
                                 const std::filesystem::path& path = default_config_path());

}  // namespace newtollm

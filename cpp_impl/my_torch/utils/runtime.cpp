#include "utils/runtime.h"

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace MyTorch {
namespace {

const nlohmann::json &shared_config() {
  static const nlohmann::json cfg = [] {
    // Repo-root relative (like assets/), with the source tree as fallback.
    const std::filesystem::path candidates[] = {
        "configs/toy_gpt2.json",
        std::filesystem::path(NEWTOLLM_SOURCE_DIR) / "configs/toy_gpt2.json"};
    for (const auto &path : candidates) {
      std::ifstream in(path);
      if (in) {
        return nlohmann::json::parse(in);
      }
    }
    throw std::runtime_error("configs/toy_gpt2.json not found");
  }();
  return cfg;
}

} // namespace

RuntimeOptions &runtime_options() {
  static RuntimeOptions options;
  return options;
}

uint64_t runtime_seed() {
  const auto &opt = runtime_options().seed;
  return opt ? *opt : shared_config().at("seed").get<uint64_t>();
}

ToyModelConfig model_config(bool allow_small) {
  const auto &j = shared_config().at(
      allow_small && runtime_options().small ? "model_small" : "model");
  ToyModelConfig cfg;
  cfg.vocab_size = j.at("vocab_size").get<int64_t>();
  cfg.emb_dim = j.at("emb_dim").get<int64_t>();
  cfg.n_layers = j.at("n_layers").get<int64_t>();
  cfg.n_heads = j.at("n_heads").get<int64_t>();
  cfg.drop_rate = j.at("drop_rate").get<double>();
  cfg.context_length = j.at("context_length").get<int64_t>();
  cfg.qkv_bias = j.at("qkv_bias").get<bool>();
  return cfg;
}

TrainConfig train_config() {
  const auto &j = shared_config().at("train");
  TrainConfig cfg;
  cfg.context_length = j.at("context_length").get<int64_t>();
  cfg.batch_size = j.at("batch_size").get<int64_t>();
  cfg.learning_rate = j.at("learning_rate").get<double>();
  cfg.weight_decay = j.at("weight_decay").get<double>();
  cfg.num_epochs = j.at("num_epochs").get<int64_t>();
  return cfg;
}

} // namespace MyTorch

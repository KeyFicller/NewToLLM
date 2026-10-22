#include "torchtwins/config.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace torchtwins {

std::filesystem::path repo_root() { return TORCHTWINS_SOURCE_DIR; }

std::filesystem::path default_config_path() { return repo_root() / "configs" / "presets.json"; }

// @twin load_preset
Preset load_preset(const std::string& name, const std::filesystem::path& path) {
  // @step read
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  const auto root = nlohmann::json::parse(in);
  const auto& presets = root.at("presets");
  // @step lookup
  if (!presets.contains(name)) {
    throw std::out_of_range("unknown preset '" + name + "'");
  }
  const auto& model = presets.at(name).at("model");
  const auto& train = presets.at(name).at("train");
  const auto& finetune = presets.at(name).at("finetune");
  // @step build
  // Designated initializers keep field order obvious next to the Python kwargs unpack.
  return Preset{
      .name = name,
      .seed = root.at("seed").get<uint64_t>(),
      .model =
          GPTConfig{
              .vocab_size = model.at("vocab_size").get<int64_t>(),
              .context_length = model.at("context_length").get<int64_t>(),
              .emb_dim = model.at("emb_dim").get<int64_t>(),
              .n_heads = model.at("n_heads").get<int64_t>(),
              .n_layers = model.at("n_layers").get<int64_t>(),
              .drop_rate = model.at("drop_rate").get<double>(),
              .qkv_bias = model.at("qkv_bias").get<bool>(),
          },
      .train =
          TrainConfig{
              .batch_size = train.at("batch_size").get<int64_t>(),
              .stride = train.at("stride").get<int64_t>(),
              .learning_rate = train.at("learning_rate").get<double>(),
              .weight_decay = train.at("weight_decay").get<double>(),
              .num_epochs = train.at("num_epochs").get<int64_t>(),
              .eval_every = train.at("eval_every").get<int64_t>(),
              .train_ratio = train.at("train_ratio").get<double>(),
          },
      .finetune =
          FinetuneConfig{
              .examples_per_class = finetune.at("examples_per_class").get<int64_t>(),
              .batch_size = finetune.at("batch_size").get<int64_t>(),
              .learning_rate = finetune.at("learning_rate").get<double>(),
              .weight_decay = finetune.at("weight_decay").get<double>(),
              .num_epochs = finetune.at("num_epochs").get<int64_t>(),
          },
  };
}

}  // namespace torchtwins

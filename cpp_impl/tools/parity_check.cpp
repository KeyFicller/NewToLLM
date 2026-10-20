// C++ half of tools/parity_check.py.
//
//   parity_check <in.pt> <out.pt>
//
// <in.pt>  (written by Python torch.save): {"config": {...}, "state": {name: tensor}, "input": LongTensor}
// <out.pt> (read by Python torch.load):    logits FloatTensor
#include "toy_model/model.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace {

std::vector<char> read_bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

c10::IValue dict_get(const c10::Dict<c10::IValue, c10::IValue> &d,
                     const std::string &key) {
  return d.at(c10::IValue(key));
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " <in.pt> <out.pt>\n";
    return 2;
  }
  const auto bytes = read_bytes(argv[1]);
  if (bytes.empty()) {
    std::cerr << "[parity] cannot read " << argv[1] << '\n';
    return 1;
  }
  const auto root = torch::pickle_load(bytes).toGenericDict();
  const auto cfg_dict = dict_get(root, "config").toGenericDict();
  const auto state = dict_get(root, "state").toGenericDict();
  const auto input = dict_get(root, "input").toTensor();

  MyTorch::ToyModelConfig cfg;
  cfg.vocab_size = dict_get(cfg_dict, "vocab_size").toInt();
  cfg.emb_dim = dict_get(cfg_dict, "emb_dim").toInt();
  cfg.n_layers = dict_get(cfg_dict, "n_layers").toInt();
  cfg.n_heads = dict_get(cfg_dict, "n_heads").toInt();
  cfg.drop_rate = dict_get(cfg_dict, "drop_rate").toDouble();
  cfg.context_length = dict_get(cfg_dict, "context_length").toInt();
  cfg.qkv_bias = dict_get(cfg_dict, "qkv_bias").toBool();

  MyTorch::ToyModel model(cfg);
  torch::NoGradGuard no_grad;
  int64_t loaded = 0;
  auto copy_from = [&](const std::string &name, torch::Tensor &dst) {
    const c10::IValue key(name);
    if (!state.contains(key)) {
      throw std::runtime_error("missing tensor in state dict: " + name);
    }
    const auto src = state.at(key).toTensor();
    TORCH_CHECK(src.sizes() == dst.sizes(), "shape mismatch for ", name);
    dst.copy_(src);
    ++loaded;
  };
  for (auto &p : model->named_parameters()) {
    copy_from(p.key(), p.value());
  }
  for (auto &b : model->named_buffers()) {
    copy_from(b.key(), b.value());
  }
  if (loaded != static_cast<int64_t>(state.size())) {
    std::cerr << "[parity] state dict has " << state.size()
              << " tensors, C++ model consumed " << loaded << '\n';
    return 1;
  }

  model->eval();
  const auto logits = model->forward(input);

  // A bare tensor keeps the file loadable with torch.load(weights_only=True).
  const auto out_bytes = torch::pickle_save(c10::IValue(logits.contiguous()));
  std::ofstream(argv[2], std::ios::binary)
      .write(out_bytes.data(), static_cast<std::streamsize>(out_bytes.size()));
  std::cout << "[parity] C++ loaded " << loaded << " tensors, logits "
            << logits.sizes() << '\n';
  return 0;
}

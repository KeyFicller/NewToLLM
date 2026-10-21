// C++ half of tools/parity.py.
//
//   parity init <preset> <seed> <out.pt>     seeded GPTModel -> tensors (named_parameters order)
//   parity forward <preset> <in.pt> <out.pt> Python's tensors + input -> logits
//   parity version                           LibTorch version
//
// Files are torch pickles: read with torch::pickle_load, written with torch::pickle_save.
#include "newtollm/config.h"
#include "newtollm/gpt.h"

#include <torch/torch.h>
#include <torch/version.h>

#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<char> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_tensors(const std::string& path, const std::vector<torch::Tensor>& tensors) {
  const auto bytes = torch::pickle_save(c10::IValue(tensors));
  std::ofstream(path, std::ios::binary)
      .write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

int init(const std::string& preset_name, uint64_t seed, const std::string& out_path) {
  const auto preset = newtollm::load_preset(preset_name).with_seed(seed);
  torch::manual_seed(preset.seed);
  const newtollm::GPTModel model(preset.model);
  std::vector<torch::Tensor> tensors;
  for (const auto& p : model->named_parameters()) {
    std::cout << p.key() << '\n';
    tensors.push_back(p.value().detach().contiguous());
  }
  write_tensors(out_path, tensors);
  return 0;
}

int forward(const std::string& preset_name, const std::string& in_path,
            const std::string& out_path) {
  // in.pt = (input_ids, [tensor for each parameter in named_parameters() order])
  const auto root = torch::pickle_load(read_bytes(in_path)).toTuple();
  const auto input = root->elements().at(0).toTensor();
  const auto weights = root->elements().at(1).toList();
  const auto preset = newtollm::load_preset(preset_name);
  newtollm::GPTModel model(preset.model);
  auto params = model->named_parameters();
  if (params.size() != weights.size()) {
    throw std::runtime_error("parameter count mismatch");
  }
  {
    const torch::NoGradGuard no_grad;
    for (size_t i = 0; i < params.size(); ++i) {
      params[i].value().copy_(weights.get(i).toTensor());
    }
  }
  model->eval();
  const torch::NoGradGuard no_grad;
  write_tensors(out_path, {model(input).contiguous()});
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::vector<std::string> args(argv + 1, argv + argc);
    if (args.size() == 1 && args[0] == "version") {
      std::cout << TORCH_VERSION << '\n';
      return 0;
    }
    if (args.size() == 4 && args[0] == "init") {
      return init(args[1], std::stoull(args[2]), args[3]);
    }
    if (args.size() == 4 && args[0] == "forward") {
      return forward(args[1], args[2], args[3]);
    }
    std::cerr
        << "usage: parity init <preset> <seed> <out.pt> | forward <preset> <in.pt> <out.pt>\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

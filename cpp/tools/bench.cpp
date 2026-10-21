// C++ half of tools/bench.py: times the very same workloads as the Python half.
//
//   bench <preset> <threads> <warmup> <iters>   -> one JSON object on stdout
//   bench version
//
// Workloads (identical shapes and seeds on both sides):
//   forward     model.eval() + NoGradGuard, logits for [batch, context] token ids
//   train_step  zero_grad + forward + cross_entropy + backward + AdamW.step
#include "newtollm/config.h"
#include "newtollm/gpt.h"
#include "newtollm/train.h"

#include <torch/torch.h>
#include <torch/version.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<double> time_ms(const std::function<void()>& fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    samples.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
            .count());
  }
  return samples;
}

double median(std::vector<double> v) {
  std::ranges::sort(v);
  const auto mid = v.size() / 2;
  return v.size() % 2 == 1 ? v[mid] : (v[mid - 1] + v[mid]) / 2.0;
}

int bench(const std::string& preset_name, int threads, int warmup, int iters) {
  torch::set_num_threads(threads);
  const auto preset = newtollm::load_preset(preset_name);
  const auto& cfg = preset.model;
  torch::manual_seed(preset.seed);
  newtollm::GPTModel model(cfg);
  const auto shape = std::vector<int64_t>{preset.train.batch_size, cfg.context_length};
  const auto inputs = torch::randint(cfg.vocab_size, shape);
  const auto targets = torch::randint(cfg.vocab_size, shape);
  torch::optim::AdamW optimizer(model->parameters(),
                                torch::optim::AdamWOptions(preset.train.learning_rate)
                                    .weight_decay(preset.train.weight_decay));

  model->eval();
  const auto forward = time_ms(
      [&] {
        const torch::NoGradGuard no_grad;
        (void)model(inputs);
      },
      warmup, iters);

  model->train();
  const auto train_step = time_ms(
      [&] {
        optimizer.zero_grad();
        newtollm::loss_batch(model, inputs, targets).backward();
        optimizer.step();
      },
      warmup, iters);

  const nlohmann::json out = {
      {"forward_ms", median(forward)},
      {"train_step_ms", median(train_step)},
      {"forward_samples", forward},
      {"train_step_samples", train_step},
  };
  std::cout << out.dump() << '\n';
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
    if (args.size() == 4) {
      return bench(args[0], std::stoi(args[1]), std::stoi(args[2]), std::stoi(args[3]));
    }
    std::cerr << "usage: bench <preset> <threads> <warmup> <iters> | bench version\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}

// C++ half of tools/benchmark.py: times ToyModel forward / train step.
//
//   benchmark [--full] [--batch B] [--tokens T] [--iters N] [--warmup W] [--threads K]
//
// Prints one JSON line: {"side":"cpp", "forward_ms":..., "train_step_ms":..., ...}
#include "toy_model/model.h"
#include "utils/runtime.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <torch/torch.h>

namespace {

template <typename F> double time_ms(F &&fn, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) fn();
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) fn();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

} // namespace

int main(int argc, char **argv) {
  bool full = false;
  int64_t batch = 2, tokens = 128;
  int iters = 5, warmup = 2, threads = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::stoll(argv[++i]); };
    if (a == "--full") full = true;
    else if (a == "--batch" && i + 1 < argc) batch = next();
    else if (a == "--tokens" && i + 1 < argc) tokens = next();
    else if (a == "--iters" && i + 1 < argc) iters = static_cast<int>(next());
    else if (a == "--warmup" && i + 1 < argc) warmup = static_cast<int>(next());
    else if (a == "--threads" && i + 1 < argc) threads = static_cast<int>(next());
    else {
      std::cerr << "unknown arg: " << a << '\n';
      return 2;
    }
  }
  if (threads > 0) torch::set_num_threads(threads);

  MyTorch::runtime_options().small = !full;
  const auto cfg = MyTorch::model_config();
  torch::manual_seed(MyTorch::runtime_seed());
  MyTorch::ToyModel model(cfg);
  auto input = torch::randint(0, cfg.vocab_size, {batch, tokens}, torch::kLong);
  auto target = torch::randint(0, cfg.vocab_size, {batch, tokens}, torch::kLong);

  model->eval();
  const double fwd = time_ms(
      [&] {
        torch::NoGradGuard ng;
        model->forward(input);
      },
      warmup, iters);

  model->train();
  torch::optim::AdamW opt(model->parameters(),
                          torch::optim::AdamWOptions(4e-4).weight_decay(0.1));
  const double step = time_ms(
      [&] {
        opt.zero_grad();
        auto logits = model->forward(input);
        auto loss = torch::nn::functional::cross_entropy(
            logits.flatten(0, 1), target.flatten());
        loss.backward();
        opt.step();
      },
      warmup, iters);

  const double ntok = static_cast<double>(batch * tokens);
  std::cout << std::fixed << std::setprecision(3) << "{\"side\": \"cpp\", \"model\": \""
            << (full ? "full" : "small") << "\", \"batch\": " << batch
            << ", \"tokens\": " << tokens << ", \"threads\": "
            << torch::get_num_threads() << ", \"forward_ms\": " << fwd
            << ", \"train_step_ms\": " << step
            << ", \"forward_tok_s\": " << ntok / fwd * 1000.0
            << ", \"train_tok_s\": " << ntok / step * 1000.0 << "}\n";
  return 0;
}

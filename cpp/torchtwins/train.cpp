// Lesson 06b - pre-training: cross-entropy, AdamW, evaluation, loss curves.
#include "torchtwins/train.h"

#include "torchtwins/report.h"

#include <ATen/CPUGeneratorImpl.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <format>
#include <fstream>

namespace torchtwins {

namespace F = torch::nn::functional;

// @twin loss_batch
torch::Tensor loss_batch(GPTModel& model, const torch::Tensor& inputs,
                         const torch::Tensor& targets) {
  // @step forward
  // inputs: [B, T] -> logits: [B, T, vocab].
  const auto logits = model(inputs);
  // @step ce
  // Flatten batch and time so CE sees [B*T, vocab] vs [B*T] — next-token LM loss.
  return F::cross_entropy(logits.flatten(0, 1), targets.flatten());
}

// @twin evaluate
double evaluate(GPTModel& model, const Windows& windows, int64_t batch_size) {
  // @step eval
  // eval() turns off dropout so val loss is deterministic across twins.
  model->eval();
  // @step loop
  double total = 0.0;
  int64_t n = 0;
  {
    const torch::NoGradGuard no_grad;
    for (const auto& [inputs, targets] : batches(windows, batch_size)) {
      total += loss_batch(model, inputs, targets).item<double>();
      ++n;
    }
  }
  // @step mean
  model->train();
  return total / static_cast<double>(std::max<int64_t>(n, 1));
}

// @twin train_model
TrainHistory train_model(GPTModel& model, const Preset& preset, const Windows& train,
                         const Windows& val) {
  const auto& cfg = preset.train;
  // @step optim
  // AdamWOptions chain matches Python kwargs; no foreach flag (LibTorch path).
  torch::optim::AdamW optimizer(
      model->parameters(),
      torch::optim::AdamWOptions(cfg.learning_rate).weight_decay(cfg.weight_decay));
  // @step state
  // Seeded Generator keeps batch order identical to Python.
  const auto generator = at::make_generator<at::CPUGeneratorImpl>(preset.seed);
  TrainHistory history;
  const auto start = std::chrono::steady_clock::now();
  model->train();
  int64_t step = 0;
  for (int64_t epoch = 0; epoch < cfg.num_epochs; ++epoch) {
    for (const auto& [inputs, targets] : batches(train, cfg.batch_size, generator)) {
      // @step step
      // zero_grad -> forward/CE -> backward -> step. Shapes stay [B, T].
      optimizer.zero_grad();
      const auto loss = loss_batch(model, inputs, targets);
      loss.backward();
      optimizer.step();
      // @step log
      ++step;
      history.tokens_seen += inputs.numel();
      history.steps.push_back(step);
      history.train_loss.push_back(loss.item<double>());
      // @step eval
      if (step % cfg.eval_every == 0) {
        history.eval_steps.push_back(step);
        history.val_loss.push_back(evaluate(model, val, cfg.batch_size));
      }
    }
  }
  // @step time
  history.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return history;
}

// @twin save_history
void save_history(const TrainHistory& history, const std::filesystem::path& path) {
  // @step dump
  // No reflection: write each field explicitly. File format matches Python asdict JSON.
  const nlohmann::json json = {
      {"steps", history.steps},
      {"train_loss", history.train_loss},
      {"eval_steps", history.eval_steps},
      {"val_loss", history.val_loss},
      {"tokens_seen", history.tokens_seen},
      {"seconds", history.seconds},
  };
  std::ofstream(path) << json.dump() << '\n';
}

// @twin run_train
void run_train(const Preset& preset, const std::optional<std::filesystem::path>& curve_path) {
  // @step setup
  const Tokenizer tokenizer;
  const auto data = load_dataset(preset, tokenizer);
  torch::manual_seed(preset.seed);
  GPTModel model(preset.model);

  // @step before
  report::section("before training");
  report::scalar("val loss", evaluate(model, data.val, preset.train.batch_size));

  // @step train
  report::section("training");
  const auto history = train_model(model, preset, data.train, data.val);
  for (size_t i = 0; i < history.eval_steps.size(); ++i) {
    const auto step = history.eval_steps[i];
    const auto train_loss = history.train_loss[static_cast<size_t>(step - 1)];
    report::text(std::format("step {:04d}", step),
                 std::format("train {:.4f}  val {:.4f}", train_loss, history.val_loss[i]));
  }
  report::count("optimizer steps", static_cast<int64_t>(history.steps.size()));
  report::count("tokens seen", history.tokens_seen);

  // @step after
  report::section("after training");
  report::scalar("val loss", evaluate(model, data.val, preset.train.batch_size));
  // unsqueeze(0) adds the batch dim Python gets from tensor([ids]).
  const auto prompt = torch::tensor(tokenizer.encode("Every effort moves you")).unsqueeze(0);
  const auto out =
      generate_greedy(model, prompt, /*max_new_tokens=*/8, preset.model.context_length);
  report::show("generated ids", out);
  // @step save
  if (curve_path) {
    save_history(history, *curve_path);
  }
}

}  // namespace torchtwins

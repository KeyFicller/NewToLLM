// Lesson 07 - fine-tuning: turn the GPT into a spam classifier.
//
// New head, frozen body, last-token classification, accuracy.
#include "newtollm/finetune.h"

#include "newtollm/modules.h"
#include "newtollm/report.h"

#include <ATen/CPUGeneratorImpl.h>

#include <algorithm>
#include <format>
#include <ranges>
#include <sstream>
#include <stdexcept>

namespace newtollm {

namespace F = torch::nn::functional;
using torch::indexing::None;
using torch::indexing::Slice;

std::filesystem::path default_sms_path() {
  return repo_root() / "assets" / "SMSSpamCollection.tsv";
}

// @twin load_sms
std::vector<Example> load_sms(const std::filesystem::path& path) {
  std::vector<Example> rows;
  std::istringstream lines(read_text(path));
  for (std::string line; std::getline(lines, line);) {
    if (!line.empty()) {
      const auto tab = line.find('\t');
      const auto label = std::ranges::find(kLabels, line.substr(0, tab)) - kLabels.begin();
      if (tab == std::string::npos || label == std::ssize(kLabels)) {
        throw std::runtime_error("bad SMS line: " + line);
      }
      rows.emplace_back(label, line.substr(tab + 1));
    }
  }
  return rows;
}

// @twin balanced
std::vector<Example> balanced(const std::vector<Example>& rows, int64_t per_class) {
  // First `per_class` ham and spam messages, interleaved ham, spam, ham, spam, ...
  const auto take = [&](int64_t label) {
    std::vector<Example> out;
    for (const auto& row :
         rows | std::views::filter([&](const Example& r) { return r.first == label; }) |
             std::views::take(per_class)) {
      out.push_back(row);
    }
    return out;
  };
  const auto ham = take(0);
  const auto spam = take(1);
  if (ham.size() != spam.size()) {
    throw std::runtime_error("not enough examples per class");
  }
  std::vector<Example> out;
  for (size_t i = 0; i < ham.size(); ++i) {
    out.push_back(ham[i]);
    out.push_back(spam[i]);
  }
  return out;
}

// @twin encode_examples
Windows encode_examples(const std::vector<Example>& examples, const Tokenizer& tokenizer,
                        int64_t context_length) {
  // Truncate to `context_length` and *left*-pad, so the last position is always real.
  const auto pad = tokenizer.end_of_text_id();
  auto inputs = torch::full({std::ssize(examples), context_length}, pad, torch::kLong);
  std::vector<int64_t> labels;
  for (int64_t row = 0; row < std::ssize(examples); ++row) {
    auto ids = tokenizer.encode(examples[static_cast<size_t>(row)].second);
    ids.resize(std::min<size_t>(ids.size(), static_cast<size_t>(context_length)));
    inputs.index_put_({row, Slice(context_length - std::ssize(ids), None)}, torch::tensor(ids));
    labels.push_back(examples[static_cast<size_t>(row)].first);
  }
  return {.inputs = inputs, .targets = torch::tensor(labels)};
}

// @twin split_examples
std::tuple<Windows, Windows, Windows> split_examples(const Windows& windows) {
  // 70% train / 10% validation / 20% test, in order (the data is already interleaved).
  const auto n = windows.size();
  const auto a = static_cast<int64_t>(0.7 * static_cast<double>(n));
  const auto b = static_cast<int64_t>(0.8 * static_cast<double>(n));
  const auto part = [&](int64_t s, int64_t e) {
    return Windows{.inputs = windows.inputs.index({Slice(s, e)}),
                   .targets = windows.targets.index({Slice(s, e)})};
  };
  return {part(0, a), part(a, b), part(b, n)};
}

// @twin freeze_body
void freeze_body(GPTModel& model, int64_t n_layers) {
  // Train only the last Transformer block, the final LayerNorm and the new head.
  const std::array<std::string, 3> trainable = {std::format("trf_blocks.{}.", n_layers - 1),
                                                "final_norm.", "out_head."};
  for (auto& item : model->named_parameters()) {
    item.value().requires_grad_(std::ranges::any_of(
        trainable, [&](const std::string& prefix) { return item.key().starts_with(prefix); }));
  }
}

// @twin class_logits
torch::Tensor class_logits(GPTModel& model, const torch::Tensor& inputs) {
  return model(inputs).index({Slice(), -1, Slice()});
}

// @twin accuracy
double accuracy(GPTModel& model, const Windows& windows, int64_t batch_size) {
  model->eval();
  int64_t correct = 0;
  int64_t total = 0;
  {
    const torch::NoGradGuard no_grad;
    for (const auto& [inputs, labels] : batches(windows, batch_size)) {
      correct += (class_logits(model, inputs).argmax(-1) == labels).sum().item<int64_t>();
      total += labels.numel();
    }
  }
  model->train();
  return static_cast<double>(correct) / static_cast<double>(std::max<int64_t>(total, 1));
}

// @twin finetune_model
FinetuneHistory finetune_model(GPTModel& model, const Preset& preset, const Windows& train,
                               const Windows& val) {
  const auto& cfg = preset.finetune;
  std::vector<torch::Tensor> params;
  std::ranges::copy_if(model->parameters(), std::back_inserter(params),
                       [](const torch::Tensor& p) { return p.requires_grad(); });
  torch::optim::AdamW optimizer(
      params, torch::optim::AdamWOptions(cfg.learning_rate).weight_decay(cfg.weight_decay));
  const auto generator = at::make_generator<at::CPUGeneratorImpl>(preset.seed);
  FinetuneHistory history;
  model->train();
  for (int64_t epoch = 0; epoch < cfg.num_epochs; ++epoch) {
    double total = 0.0;
    int64_t n = 0;
    for (const auto& [inputs, labels] : batches(train, cfg.batch_size, generator)) {
      optimizer.zero_grad();
      const auto loss = F::cross_entropy(class_logits(model, inputs), labels);
      loss.backward();
      optimizer.step();
      total += loss.item<double>();
      ++n;
    }
    history.train_loss.push_back(total / static_cast<double>(std::max<int64_t>(n, 1)));
    history.train_accuracy.push_back(accuracy(model, train, cfg.batch_size));
    history.val_accuracy.push_back(accuracy(model, val, cfg.batch_size));
  }
  return history;
}

// @twin classify
std::string classify(GPTModel& model, const Tokenizer& tokenizer, const std::string& text,
                     int64_t context_length) {
  const auto inputs = encode_examples({{0, text}}, tokenizer, context_length).inputs;
  model->eval();
  const torch::NoGradGuard no_grad;
  const auto label = class_logits(model, inputs).argmax(-1).item<int64_t>();
  return std::string(kLabels.at(static_cast<size_t>(label)));
}

// @twin run_finetune
void run_finetune(const Preset& preset) {
  const Tokenizer tokenizer;
  const auto& cfg = preset.model;
  const auto examples = balanced(load_sms(), preset.finetune.examples_per_class);
  const auto [train, val, test] =
      split_examples(encode_examples(examples, tokenizer, cfg.context_length));

  report::section("dataset");
  report::count("train examples", train.size());
  report::count("val examples", val.size());
  report::count("test examples", test.size());
  report::show("train labels", train.targets);

  report::section("new head");
  torch::manual_seed(preset.seed);
  GPTModel model(cfg);
  model->replace_head(std::ssize(kLabels));
  freeze_body(model, cfg.n_layers);
  report::count("parameters", count_parameters(*model));
  int64_t trainable = 0;
  for (const auto& p : model->parameters()) {
    trainable += p.requires_grad() ? p.numel() : 0;
  }
  report::count("trainable", trainable);
  report::scalar("test accuracy before", accuracy(model, test, preset.finetune.batch_size));

  report::section("fine-tuning");
  const auto history = finetune_model(model, preset, train, val);
  for (size_t i = 0; i < history.train_loss.size(); ++i) {
    report::text(std::format("epoch {}", i + 1),
                 std::format("loss {:.4f}  train {:.4f}  val {:.4f}", history.train_loss[i],
                             history.train_accuracy[i], history.val_accuracy[i]));
  }
  report::scalar("test accuracy after", accuracy(model, test, preset.finetune.batch_size));

  report::section("classify");
  for (const int64_t i : {-2, -1}) {  // the last two held-out test messages: one ham, one spam
    const auto& [label, text] = examples.at(examples.size() + static_cast<size_t>(i));
    const auto predicted = classify(model, tokenizer, text, cfg.context_length);
    report::text(std::format("test[{}]", i),
                 std::format("true {}  predicted {}", kLabels.at(static_cast<size_t>(label)),
                             predicted));
  }
}

}  // namespace newtollm

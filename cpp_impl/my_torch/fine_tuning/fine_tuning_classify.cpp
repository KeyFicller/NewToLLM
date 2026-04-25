#include "fine_tuning/fine_tuning_classify.h"

#include "my_torch.h"

#include <pytorch/tokenizers/tiktoken.h>
#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace MyTorch {
namespace {

constexpr const char* kTokenizerPath = "assets/gpt2.tiktoken";
constexpr const char* kSmsDatasetPath = "assets/SMSSpamCollection.tsv";
constexpr int64_t kPadTokenId = 50256;

struct LabeledText {
  int64_t label = 0;
  std::string text;
};

class ClassifyDataset : public torch::data::datasets::Dataset<ClassifyDataset> {
 public:
  ClassifyDataset(
      const std::vector<LabeledText>& rows,
      tokenizers::Tiktoken& tokenizer,
      c10::optional<int64_t> max_length)
      : max_length_(0) {
    encoded_texts_.reserve(rows.size());
    labels_.reserve(rows.size());

    for (const auto& row : rows) {
      auto encoded = tokenizer.encode(row.text, 0, 0);
      if (!encoded.ok()) {
        continue;
      }
      std::vector<int64_t> ids;
      ids.reserve(encoded->size());
      for (const auto id : *encoded) {
        ids.push_back(static_cast<int64_t>(id));
      }
      if (ids.empty()) {
        continue;
      }
      encoded_texts_.push_back(std::move(ids));
      labels_.push_back(row.label);
      max_length_ =
          std::max(max_length_, static_cast<int64_t>(encoded_texts_.back().size()));
    }

    if (max_length.has_value()) {
      max_length_ = *max_length;
      for (auto& ids : encoded_texts_) {
        if (static_cast<int64_t>(ids.size()) > max_length_) {
          ids.resize(static_cast<size_t>(max_length_));
        }
      }
    }

    for (auto& ids : encoded_texts_) {
      if (static_cast<int64_t>(ids.size()) < max_length_) {
        ids.resize(static_cast<size_t>(max_length_), kPadTokenId);
      }
    }
  }

  torch::data::Example<> get(size_t index) override {
    return {
        torch::tensor(encoded_texts_.at(index), torch::kLong),
        torch::tensor(labels_.at(index), torch::kLong)};
  }

  torch::optional<size_t> size() const override { return encoded_texts_.size(); }
  int64_t max_length() const { return max_length_; }

 private:
  std::vector<std::vector<int64_t>> encoded_texts_;
  std::vector<int64_t> labels_;
  int64_t max_length_;
};

using StackedClassifyDataset = torch::data::datasets::MapDataset<
    ClassifyDataset,
    torch::data::transforms::Stack<>>;
using TrainClassifyLoader = torch::data::StatelessDataLoader<
    StackedClassifyDataset,
    torch::data::samplers::RandomSampler>;
using EvalClassifyLoader = torch::data::StatelessDataLoader<
    StackedClassifyDataset,
    torch::data::samplers::SequentialSampler>;

std::vector<LabeledText> read_sms_rows() {
  std::ifstream file(kSmsDatasetPath);
  if (!file) {
    std::cout << "[fine_tuning] cannot open dataset: " << kSmsDatasetPath << '\n';
    return {};
  }

  std::vector<LabeledText> rows;
  std::string line;
  while (std::getline(file, line)) {
    const auto sep = line.find('\t');
    if (sep == std::string::npos) {
      continue;
    }
    const auto label = line.substr(0, sep);
    const auto text = line.substr(sep + 1);
    if (label == "ham") {
      rows.push_back({0, text});
    } else if (label == "spam") {
      rows.push_back({1, text});
    }
  }
  return rows;
}

std::vector<LabeledText> create_balanced_dataset(const std::vector<LabeledText>& rows) {
  std::vector<LabeledText> ham;
  std::vector<LabeledText> spam;
  for (const auto& row : rows) {
    (row.label == 0 ? ham : spam).push_back(row);
  }
  if (ham.empty() || spam.empty()) {
    return {};
  }

  std::mt19937 rng(123);
  std::shuffle(ham.begin(), ham.end(), rng);
  ham.resize(std::min(ham.size(), spam.size()));

  std::vector<LabeledText> balanced = spam;
  balanced.insert(balanced.end(), ham.begin(), ham.end());
  std::shuffle(balanced.begin(), balanced.end(), rng);
  return balanced;
}

void random_split(
    std::vector<LabeledText> rows,
    double train_frac,
    double val_frac,
    std::vector<LabeledText>& train_rows,
    std::vector<LabeledText>& val_rows,
    std::vector<LabeledText>& test_rows) {
  std::mt19937 rng(123);
  std::shuffle(rows.begin(), rows.end(), rng);
  const auto train_end = static_cast<size_t>(rows.size() * train_frac);
  const auto val_end = train_end + static_cast<size_t>(rows.size() * val_frac);
  train_rows.assign(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(train_end));
  val_rows.assign(
      rows.begin() + static_cast<std::ptrdiff_t>(train_end),
      rows.begin() + static_cast<std::ptrdiff_t>(val_end));
  test_rows.assign(rows.begin() + static_cast<std::ptrdiff_t>(val_end), rows.end());
}

std::unique_ptr<TrainClassifyLoader> create_train_loader(
    const std::vector<LabeledText>& rows,
    tokenizers::Tiktoken& tokenizer,
    int64_t max_length,
    int64_t batch_size) {
  auto ds = ClassifyDataset(rows, tokenizer, max_length).map(torch::data::transforms::Stack<>());
  auto options = torch::data::DataLoaderOptions()
                     .batch_size(static_cast<size_t>(batch_size))
                     .drop_last(true)
                     .workers(0);
  return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(ds), options);
}

std::unique_ptr<EvalClassifyLoader> create_eval_loader(
    const std::vector<LabeledText>& rows,
    tokenizers::Tiktoken& tokenizer,
    int64_t max_length,
    int64_t batch_size) {
  auto ds = ClassifyDataset(rows, tokenizer, max_length).map(torch::data::transforms::Stack<>());
  auto options = torch::data::DataLoaderOptions()
                     .batch_size(static_cast<size_t>(batch_size))
                     .drop_last(false)
                     .workers(0);
  return torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(ds), options);
}

torch::Tensor classify_logits(
    torch::jit::script::Module& model,
    const torch::Tensor& x,
    torch::Device device) {
  using torch::indexing::Slice;
  auto logits = model.forward({x.to(device)}).toTensor().index({Slice(), -1, Slice()});
  return logits.index({Slice(), Slice(0, 2)});
}

double calc_loss_batch(
    torch::jit::script::Module& model,
    const torch::data::Example<>& batch,
    torch::Device device) {
  auto logits = classify_logits(model, batch.data, device);
  auto target = batch.target.to(device);
  return torch::nn::functional::cross_entropy(logits, target).item<double>();
}

template <typename LoaderT>
double calc_loss_loader(
    torch::jit::script::Module& model,
    LoaderT& loader,
    torch::Device device,
    int64_t num_batches) {
  double total = 0.0;
  int64_t seen = 0;
  for (auto& batch : loader) {
    if (num_batches > 0 && seen >= num_batches) {
      break;
    }
    total += calc_loss_batch(model, batch, device);
    ++seen;
  }
  return seen == 0 ? std::numeric_limits<double>::quiet_NaN() : total / static_cast<double>(seen);
}

template <typename LoaderT>
double calc_accuracy_loader(
    torch::jit::script::Module& model,
    LoaderT& loader,
    torch::Device device,
    int64_t num_batches) {
  model.eval();
  torch::NoGradGuard no_grad;
  int64_t correct = 0;
  int64_t total = 0;
  int64_t seen = 0;
  for (auto& batch : loader) {
    if (num_batches > 0 && seen >= num_batches) {
      break;
    }
    auto logits = classify_logits(model, batch.data, device);
    auto target = batch.target.to(device);
    auto pred = torch::argmax(logits, -1);
    correct += pred.eq(target).sum().template item<int64_t>();
    total += pred.size(0);
    ++seen;
  }
  model.train();
  return total == 0 ? 0.0 : static_cast<double>(correct) / static_cast<double>(total);
}

template <typename TrainLoaderT, typename ValLoaderT>
std::pair<double, double> evaluate_model(
    torch::jit::script::Module& model,
    TrainLoaderT& train_loader,
    ValLoaderT& val_loader,
    torch::Device device,
    int64_t eval_iter) {
  model.eval();
  torch::NoGradGuard no_grad;
  const auto train_loss = calc_loss_loader(model, train_loader, device, eval_iter);
  const auto val_loss = calc_loss_loader(model, val_loader, device, eval_iter);
  model.train();
  return {train_loss, val_loss};
}

}  // namespace

void fine_tuning_classify() {
  torch::jit::script::Module model;
  try {
    model = load_saved_toy_model_jit();
  } catch (const std::exception& e) {
    std::cout << e.what() << '\n';
    return;
  }

  auto* tokenizer = get_tokenizer(kTokenizerPath);
  if (tokenizer == nullptr) {
    std::cout << "[fine_tuning] tokenizer unavailable.\n";
    return;
  }

  const auto raw_rows = read_sms_rows();
  auto balanced_rows = create_balanced_dataset(raw_rows);
  if (balanced_rows.empty()) {
    std::cout << "[fine_tuning] balanced dataset is empty.\n";
    return;
  }

  std::vector<LabeledText> train_rows;
  std::vector<LabeledText> val_rows;
  std::vector<LabeledText> test_rows;
  random_split(balanced_rows, 0.7, 0.1, train_rows, val_rows, test_rows);

  ClassifyDataset train_ds(train_rows, *tokenizer, c10::nullopt);
  const auto max_length = train_ds.max_length();
  if (max_length <= 0) {
    std::cout << "[fine_tuning] invalid sequence length from dataset.\n";
    return;
  }

  const int64_t batch_size = 8;
  auto train_loader = create_train_loader(train_rows, *tokenizer, max_length, batch_size);
  auto val_loader = create_eval_loader(val_rows, *tokenizer, max_length, batch_size);
  auto test_loader = create_eval_loader(test_rows, *tokenizer, max_length, batch_size);
  if (!train_loader || !val_loader || !test_loader) {
    std::cout << "[fine_tuning] failed to create dataloaders.\n";
    return;
  }

  std::vector<torch::Tensor> parameters;
  parameters.reserve(model.parameters().size());
  for (auto p : model.parameters()) {
    parameters.push_back(p);
  }
  for (auto& p : parameters) {
    p.set_requires_grad(true);
  }

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  model.to(device);
  model.train();

  torch::optim::AdamW optimizer(
      parameters,
      torch::optim::AdamWOptions(5e-5).weight_decay(0.1));

  const int64_t num_epochs = 5;
  const int64_t eval_freq = 50;
  const int64_t eval_iter = 5;
  int64_t global_step = -1;

  for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
    train_loader = create_train_loader(train_rows, *tokenizer, max_length, batch_size);
    for (auto& batch : *train_loader) {
      optimizer.zero_grad();
      auto logits = classify_logits(model, batch.data, device);
      auto target = batch.target.to(device);
      auto loss = torch::nn::functional::cross_entropy(logits, target);
      loss.backward();
      optimizer.step();
      ++global_step;

      if (global_step % eval_freq == 0) {
        auto train_eval_loader = create_eval_loader(train_rows, *tokenizer, max_length, batch_size);
        auto val_eval_loader = create_eval_loader(val_rows, *tokenizer, max_length, batch_size);
        const auto [train_loss, val_loss] =
            evaluate_model(model, *train_eval_loader, *val_eval_loader, device, eval_iter);
        std::cout << "[fine_tuning] Epoch " << (epoch + 1) << " Step " << global_step
                  << " Train loss " << train_loss << " Val loss " << val_loss << '\n';
      }
    }

    auto train_eval_loader = create_eval_loader(train_rows, *tokenizer, max_length, batch_size);
    auto val_eval_loader = create_eval_loader(val_rows, *tokenizer, max_length, batch_size);
    const auto train_acc = calc_accuracy_loader(model, *train_eval_loader, device, eval_iter) * 100.0;
    const auto val_acc = calc_accuracy_loader(model, *val_eval_loader, device, eval_iter) * 100.0;
    std::cout << "[fine_tuning] Epoch " << (epoch + 1) << " Train acc " << train_acc
              << "% Val acc " << val_acc << "%\n";
  }

  auto test_eval_loader = create_eval_loader(test_rows, *tokenizer, max_length, batch_size);
  const auto test_acc = calc_accuracy_loader(model, *test_eval_loader, device, 5) * 100.0;
  std::cout << "[fine_tuning] Test accuracy " << test_acc << "%\n";
}

}  // namespace MyTorch

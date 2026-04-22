#include "train/data_utils.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace MyTorch {
namespace {

bool init_tokenizer(tokenizers::Tiktoken& tokenizer, const std::string& path) {
  if (!std::filesystem::exists(path)) {
    std::cout << "[train] tokenizer file missing: " << path << '\n';
    return false;
  }
  const auto err = tokenizer.load(path);
  if (err != tokenizers::Error::Ok) {
    std::cout << "[train] failed to load tokenizer: " << path << '\n';
    return false;
  }
  return true;
}

} // namespace

tokenizers::Tiktoken* get_tokenizer(const std::string& path) {
  static std::unique_ptr<tokenizers::Tiktoken> tokenizer;
  static bool initialized = false;

  if (initialized) {
    return tokenizer.get();
  }

  initialized = true;
  tokenizer = std::make_unique<tokenizers::Tiktoken>();
  if (!init_tokenizer(*tokenizer, path)) {
    tokenizer.reset();
  }
  return tokenizer.get();
}

std::vector<int64_t> encode_text(tokenizers::Tiktoken& tokenizer,
                                 const std::string& text) {
  auto encoded = tokenizer.encode(text, 0, 0);
  if (!encoded.ok()) {
    return {};
  }

  std::vector<int64_t> ids;
  ids.reserve(encoded->size());
  for (const auto id : *encoded) {
    ids.push_back(static_cast<int64_t>(id));
  }
  return ids;
}

std::string decode_ids(tokenizers::Tiktoken& tokenizer,
                       const std::vector<int64_t>& ids) {
  std::string out;
  uint64_t prev = 0;
  bool has_prev = false;
  for (const auto id : ids) {
    const uint64_t curr = static_cast<uint64_t>(id);
    auto decoded = tokenizer.decode(has_prev ? prev : curr, curr, true);
    if (decoded.ok()) {
      out += *decoded;
    }
    prev = curr;
    has_prev = true;
  }
  return out;
}

std::string normalize_for_log(std::string text) {
  std::replace(text.begin(), text.end(), '\n', ' ');
  std::replace(text.begin(), text.end(), '\r', ' ');
  return text;
}

std::string read_text_file(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    return {};
  }
  return std::string((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
}

TextDataset::TextDataset(
    const std::vector<int64_t>& token_ids,
    int64_t max_length,
    int64_t stride) {
  std::vector<int64_t> sample_starts;
  for (int64_t i = 0; i + max_length < static_cast<int64_t>(token_ids.size());
       i += stride) {
    sample_starts.push_back(i);
  }

  features_.reserve(sample_starts.size());
  labels_.reserve(sample_starts.size());
  for (const auto start : sample_starts) {
    features_.push_back(torch::tensor(
        std::vector<int64_t>(token_ids.begin() + start,
                             token_ids.begin() + start + max_length),
        torch::kLong));
    labels_.push_back(torch::tensor(
        std::vector<int64_t>(token_ids.begin() + start + 1,
                             token_ids.begin() + start + max_length + 1),
        torch::kLong));
  }
}

torch::data::Example<> TextDataset::get(size_t index) {
  return {features_[index], labels_[index]};
}

torch::optional<size_t> TextDataset::size() const { return features_.size(); }

std::unique_ptr<TrainDataLoader> create_train_data_loader(
    const std::vector<int64_t>& token_ids,
    int64_t batch_size,
    int64_t max_length,
    int64_t stride) {
  auto dataset = TextDataset(token_ids, max_length, stride)
                     .map(torch::data::transforms::Stack<>());
  auto options = torch::data::DataLoaderOptions()
                     .batch_size(static_cast<size_t>(batch_size))
                     .drop_last(true)
                     .workers(0);
  return torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      std::move(dataset), options);
}

std::unique_ptr<EvalDataLoader> create_eval_data_loader(
    const std::vector<int64_t>& token_ids,
    int64_t batch_size,
    int64_t max_length,
    int64_t stride) {
  auto dataset = TextDataset(token_ids, max_length, stride)
                     .map(torch::data::transforms::Stack<>());
  auto options = torch::data::DataLoaderOptions()
                     .batch_size(static_cast<size_t>(batch_size))
                     .drop_last(true)
                     .workers(0);
  return torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(dataset), options);
}

double calc_loss_batch(
    ToyModel& model,
    const torch::data::Example<>& batch,
    torch::Device device) {
  auto x = batch.data.to(device);
  auto y = batch.target.to(device);
  auto logits = model->forward(x);
  auto loss = torch::nn::functional::cross_entropy(logits.flatten(0, 1), y.flatten());
  return loss.item<double>();
}

} // namespace MyTorch

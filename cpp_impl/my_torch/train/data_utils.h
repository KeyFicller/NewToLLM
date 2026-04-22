#pragma once

#include "toy_model/model.h"

#include <pytorch/tokenizers/tiktoken.h>
#include <torch/torch.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace MyTorch {

class TextDataset : public torch::data::datasets::Dataset<TextDataset> {
 public:
  TextDataset(
      const std::vector<int64_t>& token_ids,
      int64_t max_length,
      int64_t stride);

  torch::data::Example<> get(size_t index) override;
  torch::optional<size_t> size() const override;

 private:
  std::vector<torch::Tensor> features_;
  std::vector<torch::Tensor> labels_;
};

std::vector<int64_t> encode_text(tokenizers::Tiktoken& tokenizer,
                                 const std::string& text);
std::string decode_ids(tokenizers::Tiktoken& tokenizer,
                       const std::vector<int64_t>& ids);
std::string normalize_for_log(std::string text);
std::string read_text_file(const std::string& path);

using StackedTextDataset =
    torch::data::datasets::MapDataset<TextDataset, torch::data::transforms::Stack<>>;
using TrainDataLoader = torch::data::StatelessDataLoader<
    StackedTextDataset,
    torch::data::samplers::RandomSampler>;
using EvalDataLoader = torch::data::StatelessDataLoader<
    StackedTextDataset,
    torch::data::samplers::SequentialSampler>;

std::unique_ptr<TrainDataLoader> create_train_data_loader(
    const std::vector<int64_t>& token_ids,
    int64_t batch_size,
    int64_t max_length,
    int64_t stride);
std::unique_ptr<EvalDataLoader> create_eval_data_loader(
    const std::vector<int64_t>& token_ids,
    int64_t batch_size,
    int64_t max_length,
    int64_t stride);

double calc_loss_batch(
    ToyModel& model,
    const torch::data::Example<>& batch,
    torch::Device device);

template <typename LoaderT>
double calc_loss_loader(
    ToyModel& model,
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
  if (seen == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return total / static_cast<double>(seen);
}

template <typename TrainLoaderT, typename ValLoaderT>
std::pair<double, double> evaluate_model(
    ToyModel& model,
    TrainLoaderT& train_loader,
    ValLoaderT& val_loader,
    torch::Device device,
    int64_t eval_iter) {
  model->eval();
  torch::NoGradGuard no_grad;
  const double train_loss = calc_loss_loader(model, train_loader, device, eval_iter);
  const double val_loss = calc_loss_loader(model, val_loader, device, eval_iter);
  model->train();
  return {train_loss, val_loss};
}

} // namespace MyTorch

#include "my_torch.h"
#include "toy_model/model.h"
#include "train/data_utils.h"

#include <iomanip>
#include <iostream>

namespace MyTorch {
namespace {

constexpr const char* kTokenizerPath = "assets/gpt2.tiktoken";

int64_t estimate_num_batches(
    const std::vector<int64_t>& token_ids,
    int64_t max_length,
    int64_t stride,
    int64_t batch_size) {
  if (static_cast<int64_t>(token_ids.size()) <= max_length) {
    return 0;
  }
  const int64_t sample_count =
      ((static_cast<int64_t>(token_ids.size()) - max_length - 1) / stride) + 1;
  return sample_count / batch_size;
}

int64_t estimate_num_samples(
    const std::vector<int64_t>& token_ids,
    int64_t max_length,
    int64_t stride) {
  if (static_cast<int64_t>(token_ids.size()) <= max_length) {
    return 0;
  }
  return ((static_cast<int64_t>(token_ids.size()) - max_length - 1) / stride) + 1;
}

void generate_and_print_sample(ToyModel& model,
                               tokenizers::Tiktoken& tokenizer,
                               torch::Device device,
                               const std::string& start_context,
                               int64_t context_length) {
  auto prompt_ids = encode_text(tokenizer, start_context);
  if (prompt_ids.empty()) {
    std::cout << "[train] failed to encode start context.\n";
    return;
  }

  auto idx = torch::tensor(prompt_ids, torch::kLong).unsqueeze(0).to(device);
  model->eval();
  torch::NoGradGuard no_grad;
  auto out = generate_text_simple(model, idx, 50, context_length);

  auto flat = out.squeeze(0).to(torch::kLong).to(torch::kCPU);
  std::vector<int64_t> out_ids;
  out_ids.reserve(static_cast<size_t>(flat.size(0)));
  for (int64_t i = 0; i < flat.size(0); ++i) {
    out_ids.push_back(flat[i].item<int64_t>());
  }
  auto decoded = normalize_for_log(decode_ids(tokenizer, out_ids));
  std::cout << "[train] Generated text: " << decoded << '\n';
  model->train();
}

} // namespace

void train_torch() {
  torch::manual_seed(123);

  auto* tokenizer_ptr = get_tokenizer(kTokenizerPath);
  if (tokenizer_ptr == nullptr) {
    return;
  }
  auto& tokenizer = *tokenizer_ptr;

  ToyModelConfig cfg;
  cfg.context_length = 256;
  ToyModel model(cfg);

  const std::string start_context = "Every effort moves you";
  auto pre_ids = encode_text(tokenizer, start_context);
  if (!pre_ids.empty()) {
    auto pre_input = torch::tensor(pre_ids, torch::kLong).unsqueeze(0);
    model->eval();
    torch::NoGradGuard no_grad;
    auto pre_out = generate_text_simple(model, pre_input, 10, cfg.context_length);
    auto pre_flat = pre_out.squeeze(0).to(torch::kLong).to(torch::kCPU);
    std::vector<int64_t> pre_vec;
    pre_vec.reserve(static_cast<size_t>(pre_flat.size(0)));
    for (int64_t i = 0; i < pre_flat.size(0); ++i) {
      pre_vec.push_back(pre_flat[i].item<int64_t>());
    }
    std::cout << "[train] Generated text without training: "
              << normalize_for_log(decode_ids(tokenizer, pre_vec)) << '\n';
  }

  const auto text_data = read_text_file("assets/the-verdict.txt");
  if (text_data.empty()) {
    std::cout << "[train] failed to read assets/the-verdict.txt\n";
    return;
  }

  const auto split_idx = static_cast<size_t>(text_data.size() * 0.9);
  const auto train_text = text_data.substr(0, split_idx);
  const auto val_text = text_data.substr(split_idx);

  auto train_ids = encode_text(tokenizer, train_text);
  auto val_ids = encode_text(tokenizer, val_text);
  if (train_ids.empty() || val_ids.empty()) {
    std::cout << "[train] tokenization failed for training text\n";
    return;
  }

  const int64_t batch_size = 2;
  auto train_loader = create_train_data_loader(
      train_ids, batch_size, cfg.context_length, cfg.context_length);
  auto val_loader = create_eval_data_loader(
      val_ids, batch_size, cfg.context_length, cfg.context_length);

  const auto train_batches =
      estimate_num_batches(train_ids, cfg.context_length, cfg.context_length, batch_size);
  const auto val_batches =
      estimate_num_batches(val_ids, cfg.context_length, cfg.context_length, batch_size);
  const auto train_samples =
      estimate_num_samples(train_ids, cfg.context_length, cfg.context_length);
  const auto val_samples =
      estimate_num_samples(val_ids, cfg.context_length, cfg.context_length);
  std::cout << "[train] Train loader samples: " << train_samples
            << ", batches: " << train_batches
            << ", batch_size: " << batch_size << '\n';
  std::cout << "[train] Validation loader samples: " << val_samples
            << ", batches: " << val_batches
            << ", batch_size: " << batch_size << '\n';

  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  model->to(device);
  model->train();

  torch::optim::AdamW optimizer(
      model->parameters(), torch::optim::AdamWOptions(0.0004).weight_decay(0.1));

  const int64_t num_epochs = 10;
  const int64_t eval_freq = 5;
  const int64_t eval_iter = 5;
  int64_t global_step = 0;

  for (int64_t epoch = 0; epoch < num_epochs; ++epoch) {
    train_loader = create_train_data_loader(
        train_ids, batch_size, cfg.context_length, cfg.context_length);

    for (auto& batch : *train_loader) {
      optimizer.zero_grad();

      auto x = batch.data.to(device);
      auto y = batch.target.to(device);
      auto logits = model->forward(x);
      auto loss = torch::nn::functional::cross_entropy(
          logits.flatten(0, 1), y.flatten());
      loss.backward();
      optimizer.step();

      if (global_step % eval_freq == 0) {
        // Use dedicated evaluation loaders so we don't consume the active
        // training iterator.
        auto train_eval_loader = create_eval_data_loader(
            train_ids, batch_size, cfg.context_length, cfg.context_length);
        auto val_eval_loader = create_eval_data_loader(
            val_ids, batch_size, cfg.context_length, cfg.context_length);
        const auto [train_loss, val_loss] =
            evaluate_model(model, *train_eval_loader, *val_eval_loader, device, eval_iter);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[train] Epoch " << (epoch + 1) << ", Step " << global_step
                  << ", Train Loss: " << train_loss
                  << ", Val Loss: " << val_loss << '\n';
      }
      ++global_step;
    }
    generate_and_print_sample(model, tokenizer, device, start_context, cfg.context_length);
  }
}

} // namespace MyTorch

#include "newtollm/tokenizer.h"

#include "newtollm/config.h"

#include <pytorch/tokenizers/tiktoken.h>

#include <stdexcept>

namespace newtollm {

std::filesystem::path default_tokenizer_path() { return repo_root() / "assets" / "gpt2.tiktoken"; }

// @twin Tokenizer
Tokenizer::Tokenizer(const std::filesystem::path& path)
    : impl_(std::make_unique<tokenizers::Tiktoken>(
          std::string(kGPT2Pattern), std::vector<std::string>{std::string(kEndOfText)},
          /*bos_token_index=*/0, /*eos_token_index=*/0)) {
  if (impl_->load(path.string()) != tokenizers::Error::Ok) {
    throw std::runtime_error("cannot load tokenizer " + path.string());
  }
  end_of_text_id_ = static_cast<int64_t>(impl_->vocab_size()) - 1;
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int64_t> Tokenizer::encode(const std::string& text) const {
  const auto encoded = impl_->encode(text, /*bos=*/0, /*eos=*/0);
  if (!encoded.ok()) {
    throw std::runtime_error("tokenizer failed to encode text");
  }
  return {encoded->begin(), encoded->end()};
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
  std::string out;
  uint64_t prev = 0;
  for (const auto id : ids) {
    const auto token = static_cast<uint64_t>(id);
    const auto piece = impl_->decode(prev, token);
    if (piece.ok()) {
      out += *piece;
    }
    prev = token;
  }
  return out;
}

}  // namespace newtollm

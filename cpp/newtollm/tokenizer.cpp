#include "newtollm/tokenizer.h"

#include "newtollm/config.h"

#include <pytorch/tokenizers/tiktoken.h>

#include <stdexcept>

namespace newtollm {

namespace {

// A BPE token can end in the middle of a multi-byte UTF-8 character. Python's
// bytes.decode(errors="replace") turns each maximal invalid subsequence into
// U+FFFD; this does the same so both sides print identical text.
std::string replace_invalid_utf8(const std::string& bytes) {
  constexpr std::string_view kReplacement = "\xEF\xBF\xBD";
  std::string out;
  out.reserve(bytes.size());
  size_t i = 0;
  while (i < bytes.size()) {
    const auto lead = static_cast<unsigned char>(bytes[i]);
    size_t length = 0;
    unsigned char lo = 0x80;
    unsigned char hi = 0xBF;
    if (lead < 0x80) {
      length = 1;
    } else if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3;
      lo = lead == 0xE0 ? 0xA0 : 0x80;
      hi = lead == 0xED ? 0x9F : 0xBF;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4;
      lo = lead == 0xF0 ? 0x90 : 0x80;
      hi = lead == 0xF4 ? 0x8F : 0xBF;
    }
    if (length == 0) {
      out += kReplacement;
      ++i;
      continue;
    }
    size_t valid = 1;
    while (valid < length && i + valid < bytes.size()) {
      const auto next = static_cast<unsigned char>(bytes[i + valid]);
      const bool ok = valid == 1 ? (next >= lo && next <= hi) : (next >= 0x80 && next <= 0xBF);
      if (!ok) {
        break;
      }
      ++valid;
    }
    if (valid == length) {
      out.append(bytes, i, length);
    } else {
      out += kReplacement;
    }
    i += valid;
  }
  return out;
}

}  // namespace

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
  return replace_invalid_utf8(out);
}

}  // namespace newtollm

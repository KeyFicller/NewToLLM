#pragma once
// GPT-2 BPE tokenizer loaded offline from assets/gpt2.tiktoken
// (twin of py/torchtwins/tokenizer.py: same file, same regex, same special token).

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tokenizers {
class Tiktoken;
}  // namespace tokenizers

namespace torchtwins {

inline constexpr std::string_view kGPT2Pattern =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
inline constexpr std::string_view kEndOfText = "<|endoftext|>";

[[nodiscard]] std::filesystem::path default_tokenizer_path();

// @twin Tokenizer
class Tokenizer {
 public:
  explicit Tokenizer(const std::filesystem::path& path = default_tokenizer_path());
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  [[nodiscard]] std::vector<int64_t> encode(const std::string& text) const;
  [[nodiscard]] std::string decode(const std::vector<int64_t>& ids) const;
  [[nodiscard]] int64_t end_of_text_id() const { return end_of_text_id_; }

 private:
  // pImpl: hide <pytorch/tokenizers/...> behind a unique_ptr so the public header
  // stays free of third-party includes (Python just keeps a private _encoding).
  std::unique_ptr<tokenizers::Tiktoken> impl_;
  int64_t end_of_text_id_ = 0;
};

}  // namespace torchtwins

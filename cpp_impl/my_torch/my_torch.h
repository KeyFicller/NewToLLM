#pragma once

#include "my_torch_export.h"

#include <string>

namespace tokenizers {
class Tiktoken;
}

namespace MyTorch {
MY_TORCH_API void verify_torch();
MY_TORCH_API void brief_torch();
MY_TORCH_API void toy_model_torch();
MY_TORCH_API void train_torch();
MY_TORCH_API void load_model_torch();
MY_TORCH_API tokenizers::Tiktoken* get_tokenizer(const std::string& path);
} // namespace MyTorch
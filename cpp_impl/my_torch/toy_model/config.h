#pragma once

#include <cstdint>

namespace MyTorch {

struct ToyModelConfig {
  int64_t vocab_size = 50257;
  int64_t emb_dim = 768;
  int64_t n_layers = 12;
  int64_t n_heads = 12;
  double drop_rate = 0.1;
  int64_t context_length = 1024;
  bool qkv_bias = false;
};

} // namespace MyTorch

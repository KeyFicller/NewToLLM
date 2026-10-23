// Lesson 10 - OpenAI / HuggingFace GPT-2 key remapping.
#include "torchtwins/remap.h"

#include "torchtwins/gpt.h"
#include "torchtwins/modules.h"
#include "torchtwins/report.h"
#include "torchtwins/tokenizer.h"

#include <format>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torchtwins {
namespace {

torch::Tensor require_key(const std::unordered_map<std::string, torch::Tensor>& table,
                          const std::string& key) {
  const auto it = table.find(key);
  if (it == table.end()) {
    throw std::runtime_error("missing key: " + key);
  }
  return it->second;
}

}  // namespace

// @twin conv1d_to_linear
torch::Tensor conv1d_to_linear(const torch::Tensor& weight) {
  // @step transpose
  // HF Conv1D.forward is F.linear(x, weight.t(), bias), so the Linear-shaped
  // weight is exactly weight.t(). contiguous() owns storage — transpose only
  // swaps strides, and a later chunk/copy_ wants a dense (out, in) layout.
  return weight.t().contiguous();
}

// @twin linear_to_conv1d
torch::Tensor linear_to_conv1d(const torch::Tensor& weight) {
  // @step transpose
  // Inverse of conv1d_to_linear. Same .t().contiguous() — do not .view() or
  // .reshape() an (emb, 3*emb) Conv1D weight into (3*emb, emb): that keeps the
  // wrong element order and silently scrambles Q/K/V.
  return weight.t().contiguous();
}

// @twin split_qkv
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_qkv(const torch::Tensor& combined,
                                                                  int64_t emb_dim) {
  // @step check
  if (combined.size(0) != 3 * emb_dim) {
    throw std::runtime_error(std::format("combined QKV leading dim {} != 3 * emb_dim ({})",
                                         combined.size(0), 3 * emb_dim));
  }
  // @step chunk
  // After conv1d_to_linear, c_attn.weight is (3*emb, emb); bias is (3*emb,).
  // GPT-2 writes Q then K then V (not interleaved by head). chunk(3, dim=0)
  // yields three (emb, emb) / (emb,) pieces in that order.
  const auto pieces = combined.chunk(3, 0);
  return {pieces[0], pieces[1], pieces[2]};
}

// @twin merge_qkv
torch::Tensor merge_qkv(const torch::Tensor& query, const torch::Tensor& key,
                        const torch::Tensor& value) {
  // @step cat
  return torch::cat({query, key, value}, 0);
}

// @twin ours_to_hf
std::unordered_map<std::string, torch::Tensor> ours_to_hf(
    const std::unordered_map<std::string, torch::Tensor>& ours, const GPTConfig& cfg) {
  // @step require
  if (!cfg.qkv_bias) {
    throw std::runtime_error("HF GPT-2 c_attn has bias; enable qkv_bias to round-trip");
  }
  std::unordered_map<std::string, torch::Tensor> hf;
  // @step embed
  hf.emplace("transformer.wte.weight", require_key(ours, "tok_emb.weight").contiguous());
  hf.emplace("transformer.wpe.weight", require_key(ours, "pos_emb.weight").contiguous());
  // @step layers
  for (int64_t index = 0; index < cfg.n_layers; ++index) {
    const auto ours_prefix = std::format("trf_blocks.{}", index);
    const auto hf_prefix = std::format("transformer.h.{}", index);
    hf.emplace(hf_prefix + ".ln_1.weight",
               require_key(ours, ours_prefix + ".norm1.scale").contiguous());
    hf.emplace(hf_prefix + ".ln_1.bias",
               require_key(ours, ours_prefix + ".norm1.shift").contiguous());
    const auto fused_weight = merge_qkv(require_key(ours, ours_prefix + ".att.w_query.weight"),
                                        require_key(ours, ours_prefix + ".att.w_key.weight"),
                                        require_key(ours, ours_prefix + ".att.w_value.weight"));
    hf.emplace(hf_prefix + ".attn.c_attn.weight", linear_to_conv1d(fused_weight));
    const auto fused_bias = merge_qkv(require_key(ours, ours_prefix + ".att.w_query.bias"),
                                      require_key(ours, ours_prefix + ".att.w_key.bias"),
                                      require_key(ours, ours_prefix + ".att.w_value.bias"));
    hf.emplace(hf_prefix + ".attn.c_attn.bias", fused_bias.contiguous());
    hf.emplace(hf_prefix + ".attn.c_proj.weight",
               linear_to_conv1d(require_key(ours, ours_prefix + ".att.out_proj.weight")));
    hf.emplace(hf_prefix + ".attn.c_proj.bias",
               require_key(ours, ours_prefix + ".att.out_proj.bias").contiguous());
    hf.emplace(hf_prefix + ".ln_2.weight",
               require_key(ours, ours_prefix + ".norm2.scale").contiguous());
    hf.emplace(hf_prefix + ".ln_2.bias",
               require_key(ours, ours_prefix + ".norm2.shift").contiguous());
    hf.emplace(hf_prefix + ".mlp.c_fc.weight",
               linear_to_conv1d(require_key(ours, ours_prefix + ".ff.layers.0.weight")));
    hf.emplace(hf_prefix + ".mlp.c_fc.bias",
               require_key(ours, ours_prefix + ".ff.layers.0.bias").contiguous());
    hf.emplace(hf_prefix + ".mlp.c_proj.weight",
               linear_to_conv1d(require_key(ours, ours_prefix + ".ff.layers.2.weight")));
    hf.emplace(hf_prefix + ".mlp.c_proj.bias",
               require_key(ours, ours_prefix + ".ff.layers.2.bias").contiguous());
  }
  // @step head
  hf.emplace("transformer.ln_f.weight", require_key(ours, "final_norm.scale").contiguous());
  hf.emplace("transformer.ln_f.bias", require_key(ours, "final_norm.shift").contiguous());
  hf.emplace("lm_head.weight", require_key(ours, "out_head.weight").contiguous());
  return hf;
}

// @twin hf_to_ours
std::unordered_map<std::string, torch::Tensor> hf_to_ours(
    const std::unordered_map<std::string, torch::Tensor>& hf, const GPTConfig& cfg) {
  // @step require
  if (!cfg.qkv_bias) {
    throw std::runtime_error("HF GPT-2 c_attn has bias; enable qkv_bias on the target model");
  }
  std::unordered_map<std::string, torch::Tensor> ours;
  // @step embed
  ours.emplace("tok_emb.weight", require_key(hf, "transformer.wte.weight").contiguous());
  ours.emplace("pos_emb.weight", require_key(hf, "transformer.wpe.weight").contiguous());
  // @step layers
  for (int64_t index = 0; index < cfg.n_layers; ++index) {
    const auto ours_prefix = std::format("trf_blocks.{}", index);
    const auto hf_prefix = std::format("transformer.h.{}", index);
    ours.emplace(ours_prefix + ".norm1.scale",
                 require_key(hf, hf_prefix + ".ln_1.weight").contiguous());
    ours.emplace(ours_prefix + ".norm1.shift",
                 require_key(hf, hf_prefix + ".ln_1.bias").contiguous());
    const auto [query_w, key_w, value_w] = split_qkv(
        conv1d_to_linear(require_key(hf, hf_prefix + ".attn.c_attn.weight")), cfg.emb_dim);
    ours.emplace(ours_prefix + ".att.w_query.weight", query_w);
    ours.emplace(ours_prefix + ".att.w_key.weight", key_w);
    ours.emplace(ours_prefix + ".att.w_value.weight", value_w);
    const auto [query_b, key_b, value_b] =
        split_qkv(require_key(hf, hf_prefix + ".attn.c_attn.bias"), cfg.emb_dim);
    ours.emplace(ours_prefix + ".att.w_query.bias", query_b.contiguous());
    ours.emplace(ours_prefix + ".att.w_key.bias", key_b.contiguous());
    ours.emplace(ours_prefix + ".att.w_value.bias", value_b.contiguous());
    ours.emplace(ours_prefix + ".att.out_proj.weight",
                 conv1d_to_linear(require_key(hf, hf_prefix + ".attn.c_proj.weight")));
    ours.emplace(ours_prefix + ".att.out_proj.bias",
                 require_key(hf, hf_prefix + ".attn.c_proj.bias").contiguous());
    ours.emplace(ours_prefix + ".norm2.scale",
                 require_key(hf, hf_prefix + ".ln_2.weight").contiguous());
    ours.emplace(ours_prefix + ".norm2.shift",
                 require_key(hf, hf_prefix + ".ln_2.bias").contiguous());
    ours.emplace(ours_prefix + ".ff.layers.0.weight",
                 conv1d_to_linear(require_key(hf, hf_prefix + ".mlp.c_fc.weight")));
    ours.emplace(ours_prefix + ".ff.layers.0.bias",
                 require_key(hf, hf_prefix + ".mlp.c_fc.bias").contiguous());
    ours.emplace(ours_prefix + ".ff.layers.2.weight",
                 conv1d_to_linear(require_key(hf, hf_prefix + ".mlp.c_proj.weight")));
    ours.emplace(ours_prefix + ".ff.layers.2.bias",
                 require_key(hf, hf_prefix + ".mlp.c_proj.bias").contiguous());
  }
  // @step head
  ours.emplace("final_norm.scale", require_key(hf, "transformer.ln_f.weight").contiguous());
  ours.emplace("final_norm.shift", require_key(hf, "transformer.ln_f.bias").contiguous());
  // OpenAI often ties lm_head to wte and omits lm_head.weight; HF may store both.
  const auto head_it = hf.find("lm_head.weight");
  const auto& head =
      head_it == hf.end() ? require_key(hf, "transformer.wte.weight") : head_it->second;
  ours.emplace("out_head.weight", head.contiguous());
  return ours;
}

// @twin apply_state
void apply_state(torch::nn::Module& model,
                 const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step apply
  std::unordered_set<std::string> expected;
  for (const auto& item : model.named_parameters()) {
    expected.insert(item.key());
  }
  if (state.size() != expected.size()) {
    throw std::runtime_error("remap key count mismatch");
  }
  for (const auto& key : expected) {
    if (!state.contains(key)) {
      throw std::runtime_error("remap missing key: " + key);
    }
  }
  {
    const torch::NoGradGuard no_grad;
    for (auto& item : model.named_parameters()) {
      item.value().copy_(state.at(item.key()));
    }
  }
}

namespace {

// @twin max_abs_diff
double max_abs_diff(const torch::Tensor& a, const torch::Tensor& b) {
  // @step diff
  return (a - b).abs().max().item<double>();
}

}  // namespace

// @twin run_remap
void run_remap(const Preset& preset) {
  // Real GPT-2 ships c_attn/c_proj biases; the tiny preset defaults qkv_bias=false.
  GPTConfig cfg = preset.model;
  cfg.qkv_bias = true;
  cfg.drop_rate = 0.0;
  const Tokenizer tokenizer;

  // @step build
  torch::manual_seed(preset.seed);
  GPTModel model(cfg);
  model->eval();
  const auto prompt = torch::tensor(tokenizer.encode("Hello, I am")).unsqueeze(0);
  std::unordered_map<std::string, torch::Tensor> ours;
  for (const auto& item : model->named_parameters()) {
    ours.emplace(item.key(), item.value().detach());
  }

  report::section("build");
  report::count("parameters", count_parameters(*model));
  report::text("qkv_bias", "true");
  report::show("prompt ids", prompt);
  torch::Tensor fresh;
  {
    const torch::NoGradGuard no_grad;
    fresh = model(prompt);
  }
  report::show("fresh logits", fresh);

  // @step pack
  report::section("pack");
  const auto hf = ours_to_hf(ours, cfg);
  report::count("hf keys", static_cast<int64_t>(hf.size()));
  const auto c_attn = require_key(hf, "transformer.h.0.attn.c_attn.weight");
  report::show("c_attn.weight Conv1D", c_attn);
  const auto as_linear = conv1d_to_linear(c_attn);
  report::show("c_attn.weight as Linear", as_linear);
  const auto [query, key, value] = split_qkv(as_linear, cfg.emb_dim);
  (void)key;
  (void)value;
  report::show("c_attn query chunk", query);

  // @step remap
  report::section("remap");
  const auto restored_state = hf_to_ours(hf, cfg);
  report::count("ours keys", static_cast<int64_t>(restored_state.size()));
  report::text("status", "remapped");

  // @step scramble
  report::section("scramble");
  torch::manual_seed(preset.seed + 1);
  GPTModel other(cfg);
  other->eval();
  torch::Tensor scrambled;
  {
    const torch::NoGradGuard no_grad;
    scrambled = other(prompt);
  }
  report::show("scrambled logits", scrambled);

  // @step apply
  report::section("apply");
  apply_state(*other, restored_state);
  torch::Tensor restored;
  {
    const torch::NoGradGuard no_grad;
    restored = other(prompt);
  }
  report::show("restored logits", restored);
  report::scalar("max abs diff", max_abs_diff(fresh, restored));
}

}  // namespace torchtwins

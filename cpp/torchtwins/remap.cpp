// Lesson 10 - OpenAI / HuggingFace GPT-2 key remapping (twin of py/torchtwins/remap.py).

#include "torchtwins/remap.h"

#include "torchtwins/gpt.h"
#include "torchtwins/modules.h"
#include "torchtwins/report.h"
#include "torchtwins/tokenizer.h"

#include <ATen/ATen.h>

#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torchtwins {

// @twin conv1d_to_linear
torch::Tensor conv1d_to_linear(const torch::Tensor& weight) {
  // @step transpose: (in, out) -> (out, in)
  // HF Conv1D.forward is F.linear(x, weight.t(), bias), so the Linear-shaped
  // weight is exactly weight.t(). contiguous() owns storage — transpose only
  // swaps strides, and a later chunk/copy_ wants a dense (out, in) layout.
  return weight.t().contiguous();
}

// @twin linear_to_conv1d
torch::Tensor linear_to_conv1d(const torch::Tensor& weight) {
  // @step transpose: (out, in) -> (in, out)
  // Inverse of conv1d_to_linear. Same .t().contiguous() — do not .view() or
  // .reshape() an (emb, 3*emb) Conv1D weight into (3*emb, emb): that keeps the
  // wrong element order and silently scrambles Q/K/V.
  return weight.t().contiguous();
}

// @twin reshape_trap
torch::Tensor reshape_trap(const torch::Tensor& weight) {
  // @step reshape: (in, out) -> (out, in) WITHOUT transpose
  // Same *shape* as conv1d_to_linear, wrong *element order*. Used only to prove
  // the trap: learners sometimes reshape instead of transpose.
  return weight.reshape({weight.size(1), weight.size(0)});
}

// @twin split_qkv
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> split_qkv(const torch::Tensor& combined,
                                                                  int64_t emb_dim) {
  // @step check: leading size must be 3 * emb_dim
  if (combined.size(0) != 3 * emb_dim) {
    throw std::runtime_error(std::format("combined QKV leading dim {} != 3 * emb_dim ({})",
                                         combined.size(0), 3 * emb_dim));
  }
  // @step chunk: Q | K | V along dim 0
  // After conv1d_to_linear, c_attn.weight is (3*emb, emb); bias is (3*emb,).
  // GPT-2 writes Q then K then V (not interleaved by head). chunk(3, dim=0)
  // yields three (emb, emb) / (emb,) pieces in that order.
  auto parts = combined.chunk(3, 0);
  return {parts[0], parts[1], parts[2]};
}

// @twin merge_qkv
torch::Tensor merge_qkv(const torch::Tensor& query, const torch::Tensor& key,
                        const torch::Tensor& value) {
  // @step cat: stack Q|K|V back along dim 0
  return torch::cat({query, key, value}, 0);
}

// @twin to_openai_keys
std::unordered_map<std::string, torch::Tensor> to_openai_keys(
    const std::unordered_map<std::string, torch::Tensor>& hf) {
  // @step strip: transformer.foo -> foo
  static constexpr std::string_view kPrefix = "transformer.";
  std::unordered_map<std::string, torch::Tensor> out;
  for (const auto& [key, value] : hf) {
    if (key.starts_with(kPrefix)) {
      out.emplace(key.substr(kPrefix.size()), value);
    }
  }
  // @step tie: omit lm_head so load falls back to wte
  return out;
}

// @twin from_openai_keys
std::unordered_map<std::string, torch::Tensor> from_openai_keys(
    const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step passthrough: already HF-shaped
  if (state.contains("transformer.wte.weight")) {
    return state;
  }
  // @step prefix: bare OpenAI names -> transformer.*
  std::unordered_map<std::string, torch::Tensor> out;
  for (const auto& [key, value] : state) {
    if (key == "lm_head.weight") {
      out.emplace(key, value);
    } else {
      out.emplace("transformer." + key, value);
    }
  }
  return out;
}

// @twin ours_to_hf
std::unordered_map<std::string, torch::Tensor> ours_to_hf(
    const std::unordered_map<std::string, torch::Tensor>& ours, const GPTConfig& cfg) {
  // @step require: real GPT-2 has c_attn / c_proj biases
  if (!cfg.qkv_bias) {
    throw std::runtime_error("HF GPT-2 c_attn has bias; enable qkv_bias to round-trip");
  }
  std::unordered_map<std::string, torch::Tensor> hf;
  // @step embed: tok/pos -> wte/wpe (Embedding: no transpose)
  hf.emplace("transformer.wte.weight", ours.at("tok_emb.weight").contiguous());
  hf.emplace("transformer.wpe.weight", ours.at("pos_emb.weight").contiguous());
  // @step layers: each block's norms, fused attn, mlp
  for (int64_t index = 0; index < cfg.n_layers; ++index) {
    const auto ours_prefix = std::format("trf_blocks.{}", index);
    const auto hf_prefix = std::format("transformer.h.{}", index);
    hf.emplace(hf_prefix + ".ln_1.weight", ours.at(ours_prefix + ".norm1.scale").contiguous());
    hf.emplace(hf_prefix + ".ln_1.bias", ours.at(ours_prefix + ".norm1.shift").contiguous());
    const auto fused_weight = merge_qkv(ours.at(ours_prefix + ".att.w_query.weight"),
                                        ours.at(ours_prefix + ".att.w_key.weight"),
                                        ours.at(ours_prefix + ".att.w_value.weight"));
    hf.emplace(hf_prefix + ".attn.c_attn.weight", linear_to_conv1d(fused_weight));
    const auto fused_bias = merge_qkv(ours.at(ours_prefix + ".att.w_query.bias"),
                                      ours.at(ours_prefix + ".att.w_key.bias"),
                                      ours.at(ours_prefix + ".att.w_value.bias"));
    hf.emplace(hf_prefix + ".attn.c_attn.bias", fused_bias.contiguous());
    hf.emplace(hf_prefix + ".attn.c_proj.weight",
               linear_to_conv1d(ours.at(ours_prefix + ".att.out_proj.weight")));
    hf.emplace(hf_prefix + ".attn.c_proj.bias",
               ours.at(ours_prefix + ".att.out_proj.bias").contiguous());
    hf.emplace(hf_prefix + ".ln_2.weight", ours.at(ours_prefix + ".norm2.scale").contiguous());
    hf.emplace(hf_prefix + ".ln_2.bias", ours.at(ours_prefix + ".norm2.shift").contiguous());
    hf.emplace(hf_prefix + ".mlp.c_fc.weight",
               linear_to_conv1d(ours.at(ours_prefix + ".ff.layers.0.weight")));
    hf.emplace(hf_prefix + ".mlp.c_fc.bias",
               ours.at(ours_prefix + ".ff.layers.0.bias").contiguous());
    hf.emplace(hf_prefix + ".mlp.c_proj.weight",
               linear_to_conv1d(ours.at(ours_prefix + ".ff.layers.2.weight")));
    hf.emplace(hf_prefix + ".mlp.c_proj.bias",
               ours.at(ours_prefix + ".ff.layers.2.bias").contiguous());
  }
  // @step head: final LayerNorm + lm_head
  hf.emplace("transformer.ln_f.weight", ours.at("final_norm.scale").contiguous());
  hf.emplace("transformer.ln_f.bias", ours.at("final_norm.shift").contiguous());
  hf.emplace("lm_head.weight", ours.at("out_head.weight").contiguous());
  return hf;
}

// @twin hf_to_ours
std::unordered_map<std::string, torch::Tensor> hf_to_ours(
    const std::unordered_map<std::string, torch::Tensor>& hf_in, const GPTConfig& cfg) {
  // @step normalize: bare OpenAI keys -> transformer.* (no-op if already HF)
  const auto hf = from_openai_keys(hf_in);
  // @step require: need c_attn bias to fill w_query/w_key/w_value.bias
  if (!cfg.qkv_bias) {
    throw std::runtime_error("HF GPT-2 c_attn has bias; enable qkv_bias on the target model");
  }
  std::unordered_map<std::string, torch::Tensor> ours;
  // @step embed: wte/wpe -> tok/pos
  ours.emplace("tok_emb.weight", hf.at("transformer.wte.weight").contiguous());
  ours.emplace("pos_emb.weight", hf.at("transformer.wpe.weight").contiguous());
  // @step layers: split c_attn, transpose Conv1D, rename norms/mlp
  for (int64_t index = 0; index < cfg.n_layers; ++index) {
    const auto ours_prefix = std::format("trf_blocks.{}", index);
    const auto hf_prefix = std::format("transformer.h.{}", index);
    ours.emplace(ours_prefix + ".norm1.scale", hf.at(hf_prefix + ".ln_1.weight").contiguous());
    ours.emplace(ours_prefix + ".norm1.shift", hf.at(hf_prefix + ".ln_1.bias").contiguous());
    auto [query_w, key_w, value_w] =
        split_qkv(conv1d_to_linear(hf.at(hf_prefix + ".attn.c_attn.weight")), cfg.emb_dim);
    ours.emplace(ours_prefix + ".att.w_query.weight", std::move(query_w));
    ours.emplace(ours_prefix + ".att.w_key.weight", std::move(key_w));
    ours.emplace(ours_prefix + ".att.w_value.weight", std::move(value_w));
    auto [query_b, key_b, value_b] = split_qkv(hf.at(hf_prefix + ".attn.c_attn.bias"), cfg.emb_dim);
    ours.emplace(ours_prefix + ".att.w_query.bias", query_b.contiguous());
    ours.emplace(ours_prefix + ".att.w_key.bias", key_b.contiguous());
    ours.emplace(ours_prefix + ".att.w_value.bias", value_b.contiguous());
    ours.emplace(ours_prefix + ".att.out_proj.weight",
                 conv1d_to_linear(hf.at(hf_prefix + ".attn.c_proj.weight")));
    ours.emplace(ours_prefix + ".att.out_proj.bias",
                 hf.at(hf_prefix + ".attn.c_proj.bias").contiguous());
    ours.emplace(ours_prefix + ".norm2.scale", hf.at(hf_prefix + ".ln_2.weight").contiguous());
    ours.emplace(ours_prefix + ".norm2.shift", hf.at(hf_prefix + ".ln_2.bias").contiguous());
    ours.emplace(ours_prefix + ".ff.layers.0.weight",
                 conv1d_to_linear(hf.at(hf_prefix + ".mlp.c_fc.weight")));
    ours.emplace(ours_prefix + ".ff.layers.0.bias",
                 hf.at(hf_prefix + ".mlp.c_fc.bias").contiguous());
    ours.emplace(ours_prefix + ".ff.layers.2.weight",
                 conv1d_to_linear(hf.at(hf_prefix + ".mlp.c_proj.weight")));
    ours.emplace(ours_prefix + ".ff.layers.2.bias",
                 hf.at(hf_prefix + ".mlp.c_proj.bias").contiguous());
  }
  // @step head: ln_f + lm_head (fall back to wte when tied / missing)
  ours.emplace("final_norm.scale", hf.at("transformer.ln_f.weight").contiguous());
  ours.emplace("final_norm.shift", hf.at("transformer.ln_f.bias").contiguous());
  // OpenAI often ties lm_head to wte and omits lm_head.weight; HF may store both.
  const auto head_it = hf.find("lm_head.weight");
  const auto& head = head_it != hf.end() ? head_it->second : hf.at("transformer.wte.weight");
  ours.emplace("out_head.weight", head.contiguous());
  return ours;
}

// @twin apply_state
void apply_state(torch::nn::Module& model,
                 const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step apply: copy_ each Parameter by dotted name
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
  // @step diff: largest |a - b| over every element
  return (a - b).abs().max().item<double>();
}

}  // namespace

// @twin run_remap
void run_remap(const Preset& preset) {
  // Real GPT-2 ships c_attn/c_proj biases; the tiny preset defaults qkv_bias=False.
  GPTConfig cfg = preset.model;
  cfg.qkv_bias = true;
  cfg.drop_rate = 0.0;
  const Tokenizer tokenizer;

  // @step build: same-seed model (with QKV bias) and a fixed prompt
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

  // @step pack: Linear names -> synthetic HF Conv1D layout
  report::section("pack");
  auto hf = ours_to_hf(ours, cfg);
  report::count("hf keys", static_cast<int64_t>(hf.size()));
  const auto& c_attn = hf.at("transformer.h.0.attn.c_attn.weight");
  report::show("c_attn.weight Conv1D", c_attn);
  auto as_linear = conv1d_to_linear(c_attn);
  report::show("c_attn.weight as Linear", as_linear);
  auto [query, key, value] = split_qkv(as_linear, cfg.emb_dim);
  (void)key;
  (void)value;
  report::show("c_attn query chunk", query);

  // @step trap: reshape gives same shape, different values than transpose
  report::section("trap");
  auto wrong = reshape_trap(c_attn);
  report::show("reshape (wrong) as Linear", wrong);
  report::scalar("transpose vs reshape", max_abs_diff(as_linear, wrong));

  // @step remap: HF layout -> TorchTwins Linear names
  report::section("remap");
  auto restored_state = hf_to_ours(hf, cfg);
  report::count("ours keys", static_cast<int64_t>(restored_state.size()));
  report::text("status", "remapped");

  // @step scramble: new model, different seed -> different logits
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

  // @step apply: load remapped state and compare
  report::section("apply");
  apply_state(*other, restored_state);
  torch::Tensor restored;
  {
    const torch::NoGradGuard no_grad;
    restored = other(prompt);
  }
  report::show("restored logits", restored);
  // ~0 means transpose + QKV split + renames preserved every weight.
  report::scalar("max abs diff", max_abs_diff(fresh, restored));

  // @step openai: bare keys, tied lm_head->wte, same logits as tied HF
  report::section("openai");
  // Public OpenAI GPT-2 ties the output projection to wte. Mirror that before
  // stripping the HF ``transformer.`` prefix so the omitted lm_head still
  // round-trips.
  auto hf_tied = hf;
  hf_tied["lm_head.weight"] = hf_tied.at("transformer.wte.weight").contiguous();
  auto tied_state = hf_to_ours(hf_tied, cfg);
  torch::manual_seed(preset.seed + 2);
  GPTModel tied_model(cfg);
  tied_model->eval();
  apply_state(*tied_model, tied_state);
  torch::Tensor tied_logits;
  {
    const torch::NoGradGuard no_grad;
    tied_logits = tied_model(prompt);
  }
  report::show("tied logits", tied_logits);

  auto openai = to_openai_keys(hf_tied);
  report::count("openai keys", static_cast<int64_t>(openai.size()));
  report::text("lm_head omitted", openai.contains("lm_head.weight") ? "false" : "true");
  report::text("bare c_attn",
               openai.contains("h.0.attn.c_attn.weight") ? "h.0.attn.c_attn.weight" : "missing");
  auto openai_state = hf_to_ours(openai, cfg);
  report::scalar("tied out_head vs wte", max_abs_diff(openai_state.at("out_head.weight"),
                                                      openai_state.at("tok_emb.weight")));

  torch::manual_seed(preset.seed + 3);
  GPTModel openai_model(cfg);
  openai_model->eval();
  apply_state(*openai_model, openai_state);
  torch::Tensor openai_logits;
  {
    const torch::NoGradGuard no_grad;
    openai_logits = openai_model(prompt);
  }
  report::show("openai logits", openai_logits);
  report::scalar("openai max abs diff", max_abs_diff(tied_logits, openai_logits));
}

}  // namespace torchtwins

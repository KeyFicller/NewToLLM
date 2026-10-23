// Lesson 10 - OpenAI / HuggingFace GPT-2 key remapping (twin of py/torchtwins/remap.py).

#include "torchtwins/remap.h"

#include "torchtwins/gpt.h"
#include "torchtwins/modules.h"
#include "torchtwins/report.h"
#include "torchtwins/tokenizer.h"

#include <ATen/ATen.h>

#include <algorithm>
#include <ranges>

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

// @twin audit_keys
std::pair<std::vector<std::string>, std::vector<std::string>> audit_keys(
    const torch::nn::Module& model, const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step diff: expected vs state key sets
  // Remap can produce the right *shapes* and still omit or invent names.
  // Missing keys leave random init in place; unexpected keys are usually a
  // layout bug (forgot to strip a prefix, double-mapped a block, …).
  std::unordered_set<std::string> expected;
  for (const auto& item : model.named_parameters()) {
    expected.insert(item.key());
  }
  std::vector<std::string> missing;
  for (const auto& key : expected) {
    if (!state.contains(key)) {
      missing.push_back(key);
    }
  }
  std::vector<std::string> unexpected;
  for (const auto& [key, _] : state) {
    if (!expected.contains(key)) {
      unexpected.push_back(key);
    }
  }
  std::ranges::sort(missing);
  std::ranges::sort(unexpected);
  return {std::move(missing), std::move(unexpected)};
}

namespace {

std::string join_keys(const std::vector<std::string>& keys) {
  if (keys.empty()) {
    return "none";
  }
  std::string out = keys.front();
  for (size_t i = 1; i < keys.size(); ++i) {
    out += ",";
    out += keys[i];
  }
  return out;
}

}  // namespace

// @twin audit_shapes
std::vector<std::string> audit_shapes(const torch::nn::Module& model,
                                      const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step diff: names whose tensor.shape != Parameter.shape
  // Key sets can match while a forgotten Conv1D transpose (or a swapped
  // emb_dim) leaves the wrong rank/size. copy_() would raise mid-loop and
  // leave earlier parameters already overwritten — preflight first.
  std::vector<std::string> bad;
  for (const auto& item : model.named_parameters()) {
    const auto& tensor = state.at(item.key());
    if (tensor.sizes() != item.value().sizes()) {
      bad.push_back(item.key());
    }
  }
  std::ranges::sort(bad);
  return bad;
}

// @twin audit_dtypes
std::vector<std::string> audit_dtypes(const torch::nn::Module& model,
                                      const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step diff: names whose dtype or device != Parameter
  // Keys and shapes can match while a checkpoint saved in float16 (or on
  // another device) still differs. copy_() would cast / transfer mid-loop
  // and leave earlier parameters already overwritten — preflight first.
  std::vector<std::string> bad;
  for (const auto& item : model.named_parameters()) {
    const auto& tensor = state.at(item.key());
    const auto& param = item.value();
    if (tensor.dtype() != param.dtype() || tensor.device() != param.device()) {
      bad.push_back(item.key());
    }
  }
  std::ranges::sort(bad);
  return bad;
}

// @twin apply_state
void apply_state(torch::nn::Module& model,
                 const std::unordered_map<std::string, torch::Tensor>& state) {
  // @step audit: fail closed unless key sets match exactly
  const auto [missing, unexpected] = audit_keys(model, state);
  if (!missing.empty() || !unexpected.empty()) {
    throw std::runtime_error(std::format("remap keys mismatch; missing=[{}] unexpected=[{}]",
                                         join_keys(missing), join_keys(unexpected)));
  }
  // @step shapes: refuse mismatches before any copy_ (no half-loaded model)
  const auto bad_shapes = audit_shapes(model, state);
  if (!bad_shapes.empty()) {
    const std::unordered_set<std::string> bad_set(bad_shapes.begin(), bad_shapes.end());
    std::string details;
    for (const auto& item : model.named_parameters()) {
      if (!bad_set.contains(item.key())) {
        continue;
      }
      if (!details.empty()) {
        details += "; ";
      }
      const auto& got = state.at(item.key());
      const auto& want = item.value();
      details += item.key();
      details += ": got [";
      for (int64_t i = 0; i < got.dim(); ++i) {
        if (i) {
          details += ", ";
        }
        details += std::to_string(got.size(i));
      }
      details += "] want [";
      for (int64_t i = 0; i < want.dim(); ++i) {
        if (i) {
          details += ", ";
        }
        details += std::to_string(want.size(i));
      }
      details += "]";
    }
    throw std::runtime_error("remap shape mismatch; " + details);
  }
  // @step dtypes: refuse dtype/device mismatches before any copy_
  const auto bad_dtypes = audit_dtypes(model, state);
  if (!bad_dtypes.empty()) {
    throw std::runtime_error("remap dtype/device mismatch; " + join_keys(bad_dtypes));
  }
  // @step apply: copy_ each Parameter by dotted name
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

  // @step strict: missing / unexpected keys must fail closed (no partial load)
  report::section("strict");
  // Drop one real parameter: shapes of the rest still match, but apply_state
  // must refuse — otherwise that slot keeps scramble-seed garbage.
  auto incomplete = restored_state;
  incomplete.erase("tok_emb.weight");
  auto [missing, unexpected] = audit_keys(*other, incomplete);
  report::text("missing", join_keys(missing));
  report::text("missing extras", join_keys(unexpected));
  bool missing_rejected = false;
  try {
    apply_state(*other, incomplete);
  } catch (const std::runtime_error&) {
    missing_rejected = true;
  }
  report::text("missing rejects", missing_rejected ? "true" : "false");

  // Invent a key remap might emit after a bad rename: refuse before copy_.
  auto bloated = restored_state;
  bloated.emplace("not.a.real.param", restored_state.at("tok_emb.weight"));
  auto [missing2, unexpected2] = audit_keys(*other, bloated);
  report::text("unexpected", join_keys(unexpected2));
  report::text("unexpected holes", join_keys(missing2));
  bool unexpected_rejected = false;
  try {
    apply_state(*other, bloated);
  } catch (const std::runtime_error&) {
    unexpected_rejected = true;
  }
  report::text("unexpected rejects", unexpected_rejected ? "true" : "false");

  // @step shapes: same keys, wrong shape must refuse before any copy_
  report::section("shapes");
  // Simulate a remap that forgot to transpose: keep the name, corrupt the
  // leading dim. Keys still match; copy_ would fail mid-loop without preflight.
  auto warped = restored_state;
  const auto& good = warped.at("tok_emb.weight");
  warped["tok_emb.weight"] = torch::zeros({good.size(0) + 1, good.size(1)}, good.options());
  auto bad_shapes = audit_shapes(*other, warped);
  report::text("bad shapes", join_keys(bad_shapes));
  // Snapshot one untouched param so we can prove refuse-before-copy.
  torch::Tensor probe_before;
  for (const auto& item : other->named_parameters()) {
    if (item.key() == "pos_emb.weight") {
      probe_before = item.value().detach().clone();
      break;
    }
  }
  bool shape_rejected = false;
  try {
    apply_state(*other, warped);
  } catch (const std::runtime_error&) {
    shape_rejected = true;
  }
  report::text("shape rejects", shape_rejected ? "true" : "false");
  torch::Tensor probe_after;
  for (const auto& item : other->named_parameters()) {
    if (item.key() == "pos_emb.weight") {
      probe_after = item.value();
      break;
    }
  }
  report::scalar("probe unchanged", max_abs_diff(probe_before, probe_after));

  // @step dtypes: same keys+shapes, wrong dtype must refuse before any copy_
  report::section("dtypes");
  // Simulate a float16 checkpoint slot: shape matches, copy_ would silently
  // cast into the float32 Parameter and leave earlier slots already written.
  auto mistyped = restored_state;
  mistyped["tok_emb.weight"] = mistyped.at("tok_emb.weight").to(torch::kFloat16);
  auto bad_dtypes = audit_dtypes(*other, mistyped);
  report::text("bad dtypes", join_keys(bad_dtypes));
  torch::Tensor probe_dtype_before;
  for (const auto& item : other->named_parameters()) {
    if (item.key() == "pos_emb.weight") {
      probe_dtype_before = item.value().detach().clone();
      break;
    }
  }
  bool dtype_rejected = false;
  try {
    apply_state(*other, mistyped);
  } catch (const std::runtime_error&) {
    dtype_rejected = true;
  }
  report::text("dtype rejects", dtype_rejected ? "true" : "false");
  torch::Tensor probe_dtype_after;
  for (const auto& item : other->named_parameters()) {
    if (item.key() == "pos_emb.weight") {
      probe_dtype_after = item.value();
      break;
    }
  }
  report::scalar("dtype probe unchanged", max_abs_diff(probe_dtype_before, probe_dtype_after));

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

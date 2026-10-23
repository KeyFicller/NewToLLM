"""Lesson 10 - OpenAI / HuggingFace GPT-2 key remapping.

Lesson 09 carried tensors across runtimes when *names already matched*. Real
GPT-2 checkpoints (``openai-community/gpt2`` on HuggingFace) use a different
layout:

1. ``Conv1D`` stores weight as ``(in_features, out_features)``; ``nn.Linear``
   stores ``(out_features, in_features)`` — always transpose when converting.
2. Attention packs Q/K/V into one ``c_attn`` matrix of width ``3 * emb_dim``,
   which we split into ``w_query`` / ``w_key`` / ``w_value``.
3. Names differ: ``transformer.h.N.attn…`` vs ``trf_blocks.N.att…``,
   ``mlp.c_fc`` vs ``ff.layers.0``, LayerNorm ``weight``/``bias`` vs our
   ``scale``/``shift``.

CI never downloads the 124M file: we build a **synthetic** HF-shaped state dict
from our own weights (real names + Conv1D shapes), remap it back, and check
logits. The remapping math is the same as loading a real GPT-2 checkpoint.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from torchtwins import report
from torchtwins.config import GPTConfig, Preset
from torchtwins.gpt import GPTModel
from torchtwins.modules import count_parameters
from torchtwins.tokenizer import Tokenizer


# @twin conv1d_to_linear
def conv1d_to_linear(weight: torch.Tensor) -> torch.Tensor:
    # @step transpose: (in, out) -> (out, in)
    # HF Conv1D.forward is F.linear(x, weight.t(), bias), so the Linear-shaped
    # weight is exactly weight.t(). contiguous() owns storage — transpose only
    # swaps strides, and a later chunk/copy_ wants a dense (out, in) layout.
    return weight.t().contiguous()


# @twin linear_to_conv1d
def linear_to_conv1d(weight: torch.Tensor) -> torch.Tensor:
    # @step transpose: (out, in) -> (in, out)
    # Inverse of conv1d_to_linear. Same .t().contiguous() — do not .view() or
    # .reshape() an (emb, 3*emb) Conv1D weight into (3*emb, emb): that keeps the
    # wrong element order and silently scrambles Q/K/V.
    return weight.t().contiguous()


# @twin split_qkv
def split_qkv(
    combined: torch.Tensor, emb_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # @step check: leading size must be 3 * emb_dim
    if combined.shape[0] != 3 * emb_dim:
        msg = f"combined QKV leading dim {combined.shape[0]} != 3 * emb_dim ({3 * emb_dim})"
        raise ValueError(msg)
    # @step chunk: Q | K | V along dim 0
    # After conv1d_to_linear, c_attn.weight is (3*emb, emb); bias is (3*emb,).
    # GPT-2 writes Q then K then V (not interleaved by head). chunk(3, dim=0)
    # yields three (emb, emb) / (emb,) pieces in that order.
    query, key, value = combined.chunk(3, dim=0)
    return query, key, value


# @twin merge_qkv
def merge_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    # @step cat: stack Q|K|V back along dim 0
    return torch.cat([query, key, value], dim=0)


# @twin ours_to_hf
def ours_to_hf(ours: dict[str, torch.Tensor], cfg: GPTConfig) -> dict[str, torch.Tensor]:
    """Pack TorchTwins Linear weights into a HuggingFace GPT-2-shaped dict."""
    # @step require: real GPT-2 has c_attn / c_proj biases
    if not cfg.qkv_bias:
        msg = "HF GPT-2 c_attn has bias; enable qkv_bias to round-trip"
        raise ValueError(msg)
    hf: dict[str, torch.Tensor] = {}
    # @step embed: tok/pos -> wte/wpe (Embedding: no transpose)
    hf["transformer.wte.weight"] = ours["tok_emb.weight"].contiguous()
    hf["transformer.wpe.weight"] = ours["pos_emb.weight"].contiguous()
    # @step layers: each block's norms, fused attn, mlp
    for index in range(cfg.n_layers):
        ours_prefix = f"trf_blocks.{index}"
        hf_prefix = f"transformer.h.{index}"
        hf[f"{hf_prefix}.ln_1.weight"] = ours[f"{ours_prefix}.norm1.scale"].contiguous()
        hf[f"{hf_prefix}.ln_1.bias"] = ours[f"{ours_prefix}.norm1.shift"].contiguous()
        fused_weight = merge_qkv(
            ours[f"{ours_prefix}.att.w_query.weight"],
            ours[f"{ours_prefix}.att.w_key.weight"],
            ours[f"{ours_prefix}.att.w_value.weight"],
        )
        hf[f"{hf_prefix}.attn.c_attn.weight"] = linear_to_conv1d(fused_weight)
        fused_bias = merge_qkv(
            ours[f"{ours_prefix}.att.w_query.bias"],
            ours[f"{ours_prefix}.att.w_key.bias"],
            ours[f"{ours_prefix}.att.w_value.bias"],
        )
        hf[f"{hf_prefix}.attn.c_attn.bias"] = fused_bias.contiguous()
        hf[f"{hf_prefix}.attn.c_proj.weight"] = linear_to_conv1d(
            ours[f"{ours_prefix}.att.out_proj.weight"]
        )
        hf[f"{hf_prefix}.attn.c_proj.bias"] = ours[f"{ours_prefix}.att.out_proj.bias"].contiguous()
        hf[f"{hf_prefix}.ln_2.weight"] = ours[f"{ours_prefix}.norm2.scale"].contiguous()
        hf[f"{hf_prefix}.ln_2.bias"] = ours[f"{ours_prefix}.norm2.shift"].contiguous()
        hf[f"{hf_prefix}.mlp.c_fc.weight"] = linear_to_conv1d(
            ours[f"{ours_prefix}.ff.layers.0.weight"]
        )
        hf[f"{hf_prefix}.mlp.c_fc.bias"] = ours[f"{ours_prefix}.ff.layers.0.bias"].contiguous()
        hf[f"{hf_prefix}.mlp.c_proj.weight"] = linear_to_conv1d(
            ours[f"{ours_prefix}.ff.layers.2.weight"]
        )
        hf[f"{hf_prefix}.mlp.c_proj.bias"] = ours[f"{ours_prefix}.ff.layers.2.bias"].contiguous()
    # @step head: final LayerNorm + lm_head
    hf["transformer.ln_f.weight"] = ours["final_norm.scale"].contiguous()
    hf["transformer.ln_f.bias"] = ours["final_norm.shift"].contiguous()
    hf["lm_head.weight"] = ours["out_head.weight"].contiguous()
    return hf


# @twin hf_to_ours
def hf_to_ours(hf: dict[str, torch.Tensor], cfg: GPTConfig) -> dict[str, torch.Tensor]:
    """Remap a HuggingFace GPT-2 state dict into TorchTwins Linear names/shapes."""
    # @step require: need c_attn bias to fill w_query/w_key/w_value.bias
    if not cfg.qkv_bias:
        msg = "HF GPT-2 c_attn has bias; enable qkv_bias on the target model"
        raise ValueError(msg)
    ours: dict[str, torch.Tensor] = {}
    # @step embed: wte/wpe -> tok/pos
    ours["tok_emb.weight"] = hf["transformer.wte.weight"].contiguous()
    ours["pos_emb.weight"] = hf["transformer.wpe.weight"].contiguous()
    # @step layers: split c_attn, transpose Conv1D, rename norms/mlp
    for index in range(cfg.n_layers):
        ours_prefix = f"trf_blocks.{index}"
        hf_prefix = f"transformer.h.{index}"
        ours[f"{ours_prefix}.norm1.scale"] = hf[f"{hf_prefix}.ln_1.weight"].contiguous()
        ours[f"{ours_prefix}.norm1.shift"] = hf[f"{hf_prefix}.ln_1.bias"].contiguous()
        query_w, key_w, value_w = split_qkv(
            conv1d_to_linear(hf[f"{hf_prefix}.attn.c_attn.weight"]), cfg.emb_dim
        )
        ours[f"{ours_prefix}.att.w_query.weight"] = query_w
        ours[f"{ours_prefix}.att.w_key.weight"] = key_w
        ours[f"{ours_prefix}.att.w_value.weight"] = value_w
        query_b, key_b, value_b = split_qkv(hf[f"{hf_prefix}.attn.c_attn.bias"], cfg.emb_dim)
        ours[f"{ours_prefix}.att.w_query.bias"] = query_b.contiguous()
        ours[f"{ours_prefix}.att.w_key.bias"] = key_b.contiguous()
        ours[f"{ours_prefix}.att.w_value.bias"] = value_b.contiguous()
        ours[f"{ours_prefix}.att.out_proj.weight"] = conv1d_to_linear(
            hf[f"{hf_prefix}.attn.c_proj.weight"]
        )
        ours[f"{ours_prefix}.att.out_proj.bias"] = hf[f"{hf_prefix}.attn.c_proj.bias"].contiguous()
        ours[f"{ours_prefix}.norm2.scale"] = hf[f"{hf_prefix}.ln_2.weight"].contiguous()
        ours[f"{ours_prefix}.norm2.shift"] = hf[f"{hf_prefix}.ln_2.bias"].contiguous()
        ours[f"{ours_prefix}.ff.layers.0.weight"] = conv1d_to_linear(
            hf[f"{hf_prefix}.mlp.c_fc.weight"]
        )
        ours[f"{ours_prefix}.ff.layers.0.bias"] = hf[f"{hf_prefix}.mlp.c_fc.bias"].contiguous()
        ours[f"{ours_prefix}.ff.layers.2.weight"] = conv1d_to_linear(
            hf[f"{hf_prefix}.mlp.c_proj.weight"]
        )
        ours[f"{ours_prefix}.ff.layers.2.bias"] = hf[f"{hf_prefix}.mlp.c_proj.bias"].contiguous()
    # @step head: ln_f + lm_head (fall back to wte when tied / missing)
    ours["final_norm.scale"] = hf["transformer.ln_f.weight"].contiguous()
    ours["final_norm.shift"] = hf["transformer.ln_f.bias"].contiguous()
    # OpenAI often ties lm_head to wte and omits lm_head.weight; HF may store both.
    head = hf.get("lm_head.weight", hf["transformer.wte.weight"])
    ours["out_head.weight"] = head.contiguous()
    return ours


# @twin apply_state
def apply_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    # @step apply: copy_ each Parameter by dotted name
    expected = {name for name, _ in model.named_parameters()}
    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected)
    if missing or unexpected:
        msg = f"remap keys mismatch; missing={missing} unexpected={unexpected}"
        raise KeyError(msg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state[name])


# @twin max_abs_diff
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # @step diff: largest |a - b| over every element
    value: float = (a - b).abs().max().item()
    return value


# @twin run_remap
def run_remap(preset: Preset) -> None:
    # Real GPT-2 ships c_attn/c_proj biases; the tiny preset defaults qkv_bias=False.
    cfg = replace(preset.model, qkv_bias=True, drop_rate=0.0)
    tokenizer = Tokenizer()

    # @step build: same-seed model (with QKV bias) and a fixed prompt
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)
    model.eval()
    prompt = torch.tensor([tokenizer.encode("Hello, I am")])
    ours = {name: param.detach() for name, param in model.named_parameters()}

    report.section("build")
    report.count("parameters", count_parameters(model))
    report.text("qkv_bias", "true")
    report.show("prompt ids", prompt)
    with torch.no_grad():
        fresh = model(prompt)
    report.show("fresh logits", fresh)

    # @step pack: Linear names -> synthetic HF Conv1D layout
    report.section("pack")
    hf = ours_to_hf(ours, cfg)
    report.count("hf keys", len(hf))
    c_attn = hf["transformer.h.0.attn.c_attn.weight"]
    report.show("c_attn.weight Conv1D", c_attn)
    as_linear = conv1d_to_linear(c_attn)
    report.show("c_attn.weight as Linear", as_linear)
    query, _key, _value = split_qkv(as_linear, cfg.emb_dim)
    report.show("c_attn query chunk", query)

    # @step remap: HF layout -> TorchTwins Linear names
    report.section("remap")
    restored_state = hf_to_ours(hf, cfg)
    report.count("ours keys", len(restored_state))
    report.text("status", "remapped")

    # @step scramble: new model, different seed -> different logits
    report.section("scramble")
    torch.manual_seed(preset.seed + 1)
    other = GPTModel(cfg)
    other.eval()
    with torch.no_grad():
        scrambled = other(prompt)
    report.show("scrambled logits", scrambled)

    # @step apply: load remapped state and compare
    report.section("apply")
    apply_state(other, restored_state)
    with torch.no_grad():
        restored = other(prompt)
    report.show("restored logits", restored)
    # ~0 means transpose + QKV split + renames preserved every weight.
    report.scalar("max abs diff", max_abs_diff(fresh, restored))

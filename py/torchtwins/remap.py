"""Lesson 10 - OpenAI / HuggingFace GPT-2 key remapping.

Lesson 09 carried tensors across runtimes when *names already matched*. Real
GPT-2 checkpoints (``openai-community/gpt2`` on HuggingFace) use a different
layout:

1. ``Conv1D`` stores weight as ``(in_features, out_features)``; ``nn.Linear``
   stores ``(out_features, in_features)`` — always transpose when converting.
   ``reshape`` / ``view`` keep the wrong element order and scramble Q/K/V.
2. Attention packs Q/K/V into one ``c_attn`` matrix of width ``3 * emb_dim``,
   which we split into ``w_query`` / ``w_key`` / ``w_value``.
3. Names differ: HF uses a ``transformer.`` prefix (``transformer.h.N.attn…``);
   the original OpenAI release uses bare names (``h.N.attn…``, ``wte``). OpenAI
   also ties ``lm_head`` to ``wte`` and often omits ``lm_head.weight``.

CI never downloads the 124M file: we build a **synthetic** HF-shaped state dict
from our own weights (real names + Conv1D shapes), remap it back, then deepen
with (a) a reshape-vs-transpose trap, (b) an OpenAI bare-key + tied-head
round-trip, (c) a strict missing/unexpected-key audit that fails closed,
(d) a shape-mismatch preflight so a wrong Conv1D layout never half-loads,
(e) a dtype/device preflight so a silent ``copy_`` cast never half-loads, and
(f) a contiguity preflight so a transposed view (forgot ``.contiguous()``)
never lands in the Parameter storage as a strided surprise.
The remapping math is the same as loading a real GPT-2 checkpoint.
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


# @twin reshape_trap
def reshape_trap(weight: torch.Tensor) -> torch.Tensor:
    # @step reshape: (in, out) -> (out, in) WITHOUT transpose
    # Same *shape* as conv1d_to_linear, wrong *element order*. Used only to prove
    # the trap: learners sometimes reshape instead of transpose.
    return weight.reshape(weight.size(1), weight.size(0))


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


# @twin to_openai_keys
def to_openai_keys(hf: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF ``transformer.*`` (+ ``lm_head``) -> OpenAI bare keys; drop ``lm_head``.

    The public OpenAI GPT-2 release used names like ``wte.weight`` / ``h.0.…``
    with no ``transformer.`` prefix, and tied the output projection to ``wte``
    so ``lm_head.weight`` is absent.
    """
    # @step strip: transformer.foo -> foo
    out: dict[str, torch.Tensor] = {}
    for key, value in hf.items():
        if key.startswith("transformer."):
            out[key.removeprefix("transformer.")] = value
    # @step tie: omit lm_head so load falls back to wte
    return out


# @twin from_openai_keys
def from_openai_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept OpenAI bare keys or HF ``transformer.*``; always return HF names."""
    # @step passthrough: already HF-shaped
    if "transformer.wte.weight" in state:
        return dict(state)
    # @step prefix: bare OpenAI names -> transformer.*
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key == "lm_head.weight":
            out[key] = value
        else:
            out[f"transformer.{key}"] = value
    return out


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
    """Remap HF or OpenAI GPT-2 keys into TorchTwins Linear names/shapes."""
    # @step normalize: bare OpenAI keys -> transformer.* (no-op if already HF)
    hf = from_openai_keys(hf)
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


# @twin audit_keys
def audit_keys(model: nn.Module, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    # @step diff: expected vs state key sets
    # Remap can produce the right *shapes* and still omit or invent names.
    # Missing keys leave random init in place; unexpected keys are usually a
    # layout bug (forgot to strip a prefix, double-mapped a block, …).
    expected = {name for name, _ in model.named_parameters()}
    missing = sorted(expected - state.keys())
    unexpected = sorted(state.keys() - expected)
    return missing, unexpected


# @twin audit_shapes
def audit_shapes(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    # @step diff: names whose tensor.shape != Parameter.shape
    # Key sets can match while a forgotten Conv1D transpose (or a swapped
    # emb_dim) leaves the wrong rank/size. copy_() would raise mid-loop and
    # leave earlier parameters already overwritten — preflight first.
    bad: list[str] = []
    for name, param in model.named_parameters():
        tensor = state[name]
        if tuple(tensor.shape) != tuple(param.shape):
            bad.append(name)
    return sorted(bad)


# @twin audit_dtypes
def audit_dtypes(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    # @step diff: names whose dtype or device != Parameter
    # Keys and shapes can match while a checkpoint saved in float16 (or on
    # another device) still differs. ``copy_`` would cast / transfer mid-loop
    # and leave earlier parameters already overwritten — preflight first.
    bad: list[str] = []
    for name, param in model.named_parameters():
        tensor = state[name]
        if tensor.dtype != param.dtype or tensor.device != param.device:
            bad.append(name)
    return sorted(bad)



# @twin audit_contiguous
def audit_contiguous(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    # @step diff: names whose tensor is not contiguous in memory
    # Keys, shapes, and dtypes can all match while a forgotten ``.contiguous()``
    # after ``.t()`` leaves a strided view. ``copy_`` would still succeed and
    # densify into the Parameter — but the *source* state dict would keep
    # surprising strides for later save/chunk. Preflight first.
    bad: list[str] = []
    for name, _param in model.named_parameters():
        tensor = state[name]
        if not tensor.is_contiguous():
            bad.append(name)
    return sorted(bad)


# @twin apply_state
def apply_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    # @step audit: fail closed unless key sets match exactly
    missing, unexpected = audit_keys(model, state)
    if missing or unexpected:
        msg = f"remap keys mismatch; missing={missing} unexpected={unexpected}"
        raise KeyError(msg)
    # @step shapes: refuse mismatches before any copy_ (no half-loaded model)
    bad_shapes = audit_shapes(model, state)
    if bad_shapes:
        details = []
        for name, param in model.named_parameters():
            if name in bad_shapes:
                details.append(f"{name}: got {tuple(state[name].shape)} want {tuple(param.shape)}")
        msg = "remap shape mismatch; " + "; ".join(details)
        raise ValueError(msg)
    # @step dtypes: refuse dtype/device mismatches before any copy_
    bad_dtypes = audit_dtypes(model, state)
    if bad_dtypes:
        details = []
        for name, param in model.named_parameters():
            if name in bad_dtypes:
                tensor = state[name]
                details.append(
                    f"{name}: got {tensor.dtype}/{tensor.device} want {param.dtype}/{param.device}"
                )
        msg = "remap dtype/device mismatch; " + "; ".join(details)
        raise TypeError(msg)
    # @step layout: refuse non-contiguous tensors before any copy_
    bad_layout = audit_contiguous(model, state)
    if bad_layout:
        msg = "remap non-contiguous; " + ",".join(bad_layout)
        raise ValueError(msg)
    # @step apply: copy_ each Parameter by dotted name
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

    # @step trap: reshape gives same shape, different values than transpose
    report.section("trap")
    wrong = reshape_trap(c_attn)
    report.show("reshape (wrong) as Linear", wrong)
    report.scalar("transpose vs reshape", max_abs_diff(as_linear, wrong))

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

    # @step strict: missing / unexpected keys must fail closed (no partial load)
    report.section("strict")
    # Drop one real parameter: shapes of the rest still match, but apply_state
    # must refuse — otherwise that slot keeps scramble-seed garbage.
    incomplete = {name: value for name, value in restored_state.items() if name != "tok_emb.weight"}
    missing, unexpected = audit_keys(other, incomplete)
    report.text("missing", ",".join(missing) if missing else "none")
    report.text("missing extras", ",".join(unexpected) if unexpected else "none")
    missing_rejected = False
    try:
        apply_state(other, incomplete)
    except KeyError:
        missing_rejected = True
    report.text("missing rejects", "true" if missing_rejected else "false")

    # Invent a key remap might emit after a bad rename: refuse before copy_.
    bloated = dict(restored_state)
    bloated["not.a.real.param"] = restored_state["tok_emb.weight"]
    missing, unexpected = audit_keys(other, bloated)
    report.text("unexpected", ",".join(unexpected) if unexpected else "none")
    report.text("unexpected holes", ",".join(missing) if missing else "none")
    unexpected_rejected = False
    try:
        apply_state(other, bloated)
    except KeyError:
        unexpected_rejected = True
    report.text("unexpected rejects", "true" if unexpected_rejected else "false")

    # @step shapes: same keys, wrong shape must refuse before any copy_
    report.section("shapes")
    # Simulate a remap that forgot to transpose: keep the name, corrupt the
    # leading dim. Keys still match; copy_ would fail mid-loop without preflight.
    warped = dict(restored_state)
    good = warped["tok_emb.weight"]
    warped["tok_emb.weight"] = torch.zeros(good.shape[0] + 1, good.shape[1], dtype=good.dtype)
    bad_shapes = audit_shapes(other, warped)
    report.text("bad shapes", ",".join(bad_shapes) if bad_shapes else "none")
    # Snapshot one untouched param so we can prove refuse-before-copy.
    probe_before = other.pos_emb.weight.detach().clone()
    shape_rejected = False
    try:
        apply_state(other, warped)
    except ValueError:
        shape_rejected = True
    report.text("shape rejects", "true" if shape_rejected else "false")
    report.scalar("probe unchanged", max_abs_diff(probe_before, other.pos_emb.weight))

    # @step dtypes: same keys+shapes, wrong dtype must refuse before any copy_
    report.section("dtypes")
    # Simulate a float16 checkpoint slot: shape matches, copy_ would silently
    # cast into the float32 Parameter and leave earlier slots already written.
    mistyped = dict(restored_state)
    mistyped["tok_emb.weight"] = mistyped["tok_emb.weight"].to(torch.float16)
    bad_dtypes = audit_dtypes(other, mistyped)
    report.text("bad dtypes", ",".join(bad_dtypes) if bad_dtypes else "none")
    probe_dtype_before = other.pos_emb.weight.detach().clone()
    dtype_rejected = False
    try:
        apply_state(other, mistyped)
    except TypeError:
        dtype_rejected = True
    report.text("dtype rejects", "true" if dtype_rejected else "false")
    report.scalar("dtype probe unchanged", max_abs_diff(probe_dtype_before, other.pos_emb.weight))

    # @step layout: same keys+shapes+dtypes, non-contiguous must refuse before copy_
    report.section("layout")
    # Simulate a remap that did ``.t()`` but forgot ``.contiguous()``: shape and
    # dtype match, yet storage is a strided view. copy_ would densify mid-loop.
    strided = dict(restored_state)
    strided["tok_emb.weight"] = strided["tok_emb.weight"].t().contiguous().t()
    bad_layout = audit_contiguous(other, strided)
    report.text("bad layout", ",".join(bad_layout) if bad_layout else "none")
    report.text(
        "tok_emb contiguous",
        "true" if strided["tok_emb.weight"].is_contiguous() else "false",
    )
    probe_layout_before = other.pos_emb.weight.detach().clone()
    layout_rejected = False
    try:
        apply_state(other, strided)
    except ValueError:
        layout_rejected = True
    report.text("layout rejects", "true" if layout_rejected else "false")
    report.scalar(
        "layout probe unchanged", max_abs_diff(probe_layout_before, other.pos_emb.weight)
    )

    # @step openai: bare keys, tied lm_head->wte, same logits as tied HF
    report.section("openai")
    # Public OpenAI GPT-2 ties the output projection to wte. Mirror that before
    # stripping the HF ``transformer.`` prefix so the omitted lm_head still
    # round-trips.
    hf_tied = dict(hf)
    hf_tied["lm_head.weight"] = hf_tied["transformer.wte.weight"].contiguous()
    tied_state = hf_to_ours(hf_tied, cfg)
    torch.manual_seed(preset.seed + 2)
    tied_model = GPTModel(cfg)
    tied_model.eval()
    apply_state(tied_model, tied_state)
    with torch.no_grad():
        tied_logits = tied_model(prompt)
    report.show("tied logits", tied_logits)

    openai = to_openai_keys(hf_tied)
    report.count("openai keys", len(openai))
    report.text("lm_head omitted", "true" if "lm_head.weight" not in openai else "false")
    report.text(
        "bare c_attn", "h.0.attn.c_attn.weight" if "h.0.attn.c_attn.weight" in openai else "missing"
    )
    openai_state = hf_to_ours(openai, cfg)
    report.scalar(
        "tied out_head vs wte",
        max_abs_diff(openai_state["out_head.weight"], openai_state["tok_emb.weight"]),
    )

    torch.manual_seed(preset.seed + 3)
    openai_model = GPTModel(cfg)
    openai_model.eval()
    apply_state(openai_model, openai_state)
    with torch.no_grad():
        openai_logits = openai_model(prompt)
    report.show("openai logits", openai_logits)
    report.scalar("openai max abs diff", max_abs_diff(tied_logits, openai_logits))

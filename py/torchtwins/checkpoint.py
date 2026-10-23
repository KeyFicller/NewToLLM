"""Lesson 08 - checkpoints: save weights, reload into a fresh model.

Same-runtime round-trip only. A Python ``state_dict`` pickle/zip is not a
LibTorch module archive (and the reverse), so cross-language load is a follow-up.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from torchtwins import report
from torchtwins.config import Preset
from torchtwins.gpt import GPTModel
from torchtwins.modules import count_parameters
from torchtwins.tokenizer import Tokenizer


# @twin save_weights
def save_weights(model: GPTModel, path: Path) -> None:
    # @step dump: write the state_dict with torch.save
    # state_dict is an OrderedDict of parameter *and* buffer tensors (e.g. the
    # causal mask). torch.save packs it in Python's zip+pickle container.
    # LibTorch's torch::save writes a different archive — the files do not mix.
    torch.save(model.state_dict(), path)


# @twin load_weights
def load_weights(model: GPTModel, path: Path) -> None:
    # @step read: load tensors only (no arbitrary pickle objects)
    # weights_only=True keeps the load to tensor storage; map_location pins CPU.
    state = torch.load(path, map_location="cpu", weights_only=True)
    # @step apply: copy each tensor into this module by name
    # Keys must match named_parameters() / named_buffers() exactly.
    model.load_state_dict(state)


# @twin tensor_count
def tensor_count(model: GPTModel) -> int:
    # @step count: parameter tensors that must round-trip
    # Causal masks use register_buffer(..., persistent=False), so they are
    # *not* in state_dict — rebuilt from context_length on construction.
    # We count parameters only so the transcript matches LibTorch (which has
    # no persistent=False and would otherwise include the two masks).
    return sum(1 for _ in model.parameters())


# @twin max_abs_diff
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # @step diff: largest |a - b| over every element
    value: float = (a - b).abs().max().item()
    return value


# @twin run_checkpoint
def run_checkpoint(preset: Preset) -> None:
    cfg = preset.model
    tokenizer = Tokenizer()

    # @step build: same-seed model and a fixed prompt
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)
    model.eval()
    prompt = torch.tensor([tokenizer.encode("Hello, I am")])

    report.section("build")
    report.count("parameters", count_parameters(model))
    report.count("parameter tensors", tensor_count(model))
    report.show("prompt ids", prompt)
    with torch.no_grad():
        fresh = model(prompt)
    report.show("fresh logits", fresh)

    # @step save: write a temp checkpoint (path not printed — formats differ)
    report.section("save")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weights.pt"
        save_weights(model, path)
        report.text("status", "saved")

        # @step scramble: new model, different seed -> different logits
        report.section("scramble")
        torch.manual_seed(preset.seed + 1)
        other = GPTModel(cfg)
        other.eval()
        with torch.no_grad():
            scrambled = other(prompt)
        report.show("scrambled logits", scrambled)

        # @step load: restore saved weights into the scrambled model
        report.section("load")
        load_weights(other, path)
        with torch.no_grad():
            restored = other(prompt)
        report.show("restored logits", restored)
        # After a correct round-trip this is ~0; proves the file carried the weights.
        report.scalar("max abs diff", max_abs_diff(fresh, restored))

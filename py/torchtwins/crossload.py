"""Lesson 09 - portable weight bridge across Python and C++.

Lesson 08's archives do not mix: ``torch.save(state_dict)`` is zip+pickle of a
module dict; ``torch::save(model)`` is a LibTorch OutputArchive. This lesson
uses a third layout both runtimes already speak — a pickle of
``(parameter_names, parameter_tensors)`` in ``named_parameters()`` order.

That is the same dialect ``tools/parity.py`` uses to shove Python weights into
the C++ model. Same-runtime round-trip here; a file written by one side loads
on the other (see comments for the ``weights_only`` pitfall on the C++→Python
direction). OpenAI / HuggingFace key remapping (Conv1D transpose, combined QKV)
is the next slice — not this one.
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


# @twin export_portable
def export_portable(model: GPTModel, path: Path) -> None:
    # @step collect: names + contiguous parameter tensors
    # named_parameters() order is identical on both twins (parity gate). Names
    # travel with the tensors so a loader can match by key, not by position —
    # that is what an OpenAI/HF remap will need later.
    names = [name for name, _ in model.named_parameters()]
    tensors = [param.detach().contiguous() for _, param in model.named_parameters()]
    # @step dump: pickle the (names, tensors) pair
    # Default torch.save uses a zip container (bytes start with PK). LibTorch
    # torch::pickle_load reads that zip; torch::save(Module) does NOT produce
    # this layout. Buffers (causal masks) are omitted on purpose — rebuilt from
    # context_length on both sides.
    torch.save((names, tensors), path)


# @twin import_portable
def import_portable(model: GPTModel, path: Path) -> None:
    # @step read: load the portable pair
    # weights_only=True is safe for files *this* process wrote (list[str] +
    # tensors). A file from LibTorch pickle_save carries TorchScript type tags
    # that the weights_only allow-list rejects — use weights_only=False only for
    # a trusted C++ export (see tools/parity.py).
    names, tensors = torch.load(path, map_location="cpu", weights_only=True)
    # @step apply: copy each tensor into this module by name
    table = dict(zip(names, tensors, strict=True))
    expected = {name for name, _ in model.named_parameters()}
    missing = sorted(expected - table.keys())
    unexpected = sorted(table.keys() - expected)
    if missing or unexpected:
        msg = f"portable checkpoint keys mismatch; missing={missing} unexpected={unexpected}"
        raise KeyError(msg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(table[name])


# @twin param_names
def param_names(model: GPTModel) -> list[str]:
    # @step list: every trainable tensor's dotted path
    return [name for name, _ in model.named_parameters()]


# @twin logits_diff
def logits_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    # @step diff: largest |a - b| over every element
    value: float = (a - b).abs().max().item()
    return value


# @twin run_crossload
def run_crossload(preset: Preset) -> None:
    cfg = preset.model
    tokenizer = Tokenizer()

    # @step build: same-seed model and a fixed prompt
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)
    model.eval()
    prompt = torch.tensor([tokenizer.encode("Hello, I am")])
    names = param_names(model)

    report.section("build")
    report.count("parameters", count_parameters(model))
    report.count("parameter tensors", len(names))
    # First name proves the key string crossed the wire, not just a bag of floats.
    report.text("first parameter", names[0])
    report.show("prompt ids", prompt)
    with torch.no_grad():
        fresh = model(prompt)
    report.show("fresh logits", fresh)

    # @step export: write the portable (names, tensors) pickle
    report.section("export")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "portable.pt"
        export_portable(model, path)
        # Label the layout explicitly — lesson 08's Module archive is a different animal.
        report.text("format", "portable pickle (names, tensors)")
        report.text("status", "exported")

        # @step scramble: new model, different seed -> different logits
        report.section("scramble")
        torch.manual_seed(preset.seed + 1)
        other = GPTModel(cfg)
        other.eval()
        with torch.no_grad():
            scrambled = other(prompt)
        report.show("scrambled logits", scrambled)

        # @step import: restore via the portable bridge
        report.section("import")
        import_portable(other, path)
        with torch.no_grad():
            restored = other(prompt)
        report.show("restored logits", restored)
        # ~0 means the portable file carried every parameter by name.
        report.scalar("max abs diff", logits_diff(fresh, restored))

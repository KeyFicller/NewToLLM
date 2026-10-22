"""Lesson 06a - data pipeline: tokens -> train/val split -> sliding windows -> batches.

Every step is a tensor op that exists verbatim in LibTorch, so both sides build the
*same* windows in the *same* order (this is what fixed the old 85-vs-75 step mismatch).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from torchtwins import report
from torchtwins.config import REPO_ROOT, Preset
from torchtwins.tokenizer import Tokenizer

DEFAULT_TEXT_PATH = REPO_ROOT / "assets" / "the-verdict.txt"


# @twin Windows
@dataclass(frozen=True)
class Windows:
    # @step fields: input and target tensors
    # inputs: [num_windows, context_length] int64 token ids.
    # targets: same shape of *next* tokens for LM, or class labels [n] in lesson 07.
    inputs: torch.Tensor
    targets: torch.Tensor

    # @step len: number of windows
    def __len__(self) -> int:
        return int(self.inputs.shape[0])


# @twin Dataset
@dataclass(frozen=True)
class Dataset:
    # @step fields: train and val window packs
    train: Windows
    val: Windows


# @twin split_ids
def split_ids(ids: torch.Tensor, train_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    # Split on *tokens*, not characters: str length (code points) and std::string
    # length (bytes) disagree on non-ASCII text; token counts never do.
    # @step cut: cut index from token count
    cut = int(train_ratio * ids.numel())
    # @step slice: two contiguous segments
    # ids is 1-D [N]; train gets [:cut], val gets [cut:].
    return ids[:cut], ids[cut:]


# @twin make_windows
def make_windows(ids: torch.Tensor, context_length: int, stride: int) -> Windows:
    # @step unfold: sliding windows
    # ids[:-1] / ids[1:] shift by one so each row predicts the next token.
    # unfold(0, context_length, stride) -> [num_windows, context_length].
    # With stride == context_length the windows tile without overlap (tiny preset).
    inputs = ids[:-1].unfold(0, context_length, stride)
    targets = ids[1:].unfold(0, context_length, stride)
    # @step pack: wrap as Windows
    # contiguous() so later fancy indexing / LibTorch views stay well-defined.
    return Windows(inputs.contiguous(), targets.contiguous())


# @twin batches
def batches(
    windows: Windows, batch_size: int, generator: torch.Generator | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield full batches only (drop_last); shuffle with ``randperm`` when given a generator."""
    # @step order: optional shuffle
    # Same seeded Generator on both sides -> identical epoch order (parity gate).
    n = len(windows)
    order = torch.arange(n) if generator is None else torch.randperm(n, generator=generator)
    # @step slice: gather full batches
    # Each yield is ([B, T], [B, T]) for LM, or ([B, T], [B]) for classification.
    for start in range(0, n - batch_size + 1, batch_size):
        idx = order[start : start + batch_size]
        yield windows.inputs[idx], windows.targets[idx]


# @twin load_dataset
def load_dataset(preset: Preset, tokenizer: Tokenizer, path: Path = DEFAULT_TEXT_PATH) -> Dataset:
    # @step encode: read text and tokenize
    # Flat 1-D int64 id stream shared by both languages.
    ids = torch.tensor(tokenizer.encode(path.read_text(encoding="utf-8")))
    # @step split: train / val token cut
    train_ids, val_ids = split_ids(ids, preset.train.train_ratio)
    # @step window: build both window packs
    ctx, stride = preset.model.context_length, preset.train.stride
    return Dataset(make_windows(train_ids, ctx, stride), make_windows(val_ids, ctx, stride))


# @twin run_data
def run_data(preset: Preset) -> None:
    tokenizer = Tokenizer()
    # @step tokens: token counts
    report.section("tokens")
    ids = torch.tensor(tokenizer.encode(DEFAULT_TEXT_PATH.read_text(encoding="utf-8")))
    report.show("ids", ids)
    train_ids, val_ids = split_ids(ids, preset.train.train_ratio)
    report.count("train tokens", train_ids.numel())
    report.count("val tokens", val_ids.numel())

    # @step windows: window shapes
    report.section("windows")
    data = load_dataset(preset, tokenizer)
    report.show("train inputs", data.train.inputs)
    report.show("train targets", data.train.targets)
    report.count("train windows", len(data.train))
    report.count("val windows", len(data.val))

    # @step batches: batch counts
    report.section("batches")
    generator = torch.Generator().manual_seed(preset.seed)
    train_batches = list(batches(data.train, preset.train.batch_size, generator))
    report.count("train batches per epoch", len(train_batches))
    report.count("val batches", sum(1 for _ in batches(data.val, preset.train.batch_size)))
    report.show("first batch inputs", train_batches[0][0])

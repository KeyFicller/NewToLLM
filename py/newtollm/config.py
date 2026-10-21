"""Typed view of ``configs/presets.json`` (shared with the C++ side)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "presets.json"


# @twin GPTConfig
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool


# @twin TrainConfig
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int
    stride: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    eval_every: int
    train_ratio: float


# @twin Preset
@dataclass(frozen=True)
class Preset:
    name: str
    seed: int
    model: GPTConfig
    train: TrainConfig

    def with_seed(self, seed: int) -> Preset:
        return replace(self, seed=seed)


# @twin load_preset
def load_preset(name: str, path: Path = DEFAULT_CONFIG_PATH) -> Preset:
    root: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    presets: dict[str, Any] = root["presets"]
    if name not in presets:
        msg = f"unknown preset '{name}' (known: {', '.join(presets)})"
        raise KeyError(msg)
    entry = presets[name]
    return Preset(
        name=name,
        seed=int(root["seed"]),
        model=GPTConfig(**entry["model"]),
        train=TrainConfig(**entry["train"]),
    )

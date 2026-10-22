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
    # @step fields: 模型超参字段
    # frozen=True makes instances immutable, like a C++ aggregate you never mutate in place.
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
    # @step fields: 预训练超参字段
    batch_size: int
    stride: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    eval_every: int
    train_ratio: float


# @twin FinetuneConfig
@dataclass(frozen=True)
class FinetuneConfig:
    # @step fields: 分类微调超参字段
    examples_per_class: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int


# @twin Preset
@dataclass(frozen=True)
class Preset:
    # @step fields: 名字 + 全局种子 + 三份子配置
    name: str
    seed: int
    model: GPTConfig
    train: TrainConfig
    finetune: FinetuneConfig

    # @step with_seed: 只替换种子并返回新 Preset
    def with_seed(self, seed: int) -> Preset:
        return replace(self, seed=seed)


# @twin load_preset
def load_preset(name: str, path: Path = DEFAULT_CONFIG_PATH) -> Preset:
    # @step read: 读 configs/presets.json
    root: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    presets: dict[str, Any] = root["presets"]
    # @step lookup: 按名字取条目 未知则报错
    if name not in presets:
        msg = f"unknown preset '{name}' (known: {', '.join(presets)})"
        raise KeyError(msg)
    entry = presets[name]
    # @step build: 拆成 GPTConfig / TrainConfig / FinetuneConfig
    # Field names match the JSON keys 1:1 so both sides can unpack without a schema layer.
    return Preset(
        name=name,
        seed=int(root["seed"]),
        model=GPTConfig(**entry["model"]),
        train=TrainConfig(**entry["train"]),
        finetune=FinetuneConfig(**entry["finetune"]),
    )

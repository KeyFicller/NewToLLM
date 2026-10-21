"""Lesson 06b - pre-training: cross-entropy, AdamW, evaluation, loss curves."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F  # noqa: N812

from newtollm import report
from newtollm.config import Preset
from newtollm.data import Windows, batches, load_dataset
from newtollm.gpt import GPTModel, generate_greedy
from newtollm.tokenizer import Tokenizer


# @twin TrainHistory
@dataclass
class TrainHistory:
    steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    eval_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    tokens_seen: int = 0
    seconds: float = 0.0


# @twin loss_batch
def loss_batch(model: GPTModel, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


# @twin evaluate
def evaluate(model: GPTModel, windows: Windows, batch_size: int) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for inputs, targets in batches(windows, batch_size):
            total += loss_batch(model, inputs, targets).item()
            n += 1
    model.train()
    return total / max(n, 1)


# @twin train_model
def train_model(model: GPTModel, preset: Preset, train: Windows, val: Windows) -> TrainHistory:
    cfg = preset.train
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, foreach=False
    )
    generator = torch.Generator().manual_seed(preset.seed)
    history = TrainHistory()
    start = time.perf_counter()
    model.train()
    step = 0
    for _ in range(cfg.num_epochs):
        for inputs, targets in batches(train, cfg.batch_size, generator):
            optimizer.zero_grad()
            loss = loss_batch(model, inputs, targets)
            loss.backward()
            optimizer.step()
            step += 1
            history.tokens_seen += inputs.numel()
            history.steps.append(step)
            history.train_loss.append(loss.item())
            if step % cfg.eval_every == 0:
                history.eval_steps.append(step)
                history.val_loss.append(evaluate(model, val, cfg.batch_size))
    history.seconds = time.perf_counter() - start
    return history


# @twin save_history
def save_history(history: TrainHistory, path: Path) -> None:
    path.write_text(json.dumps(asdict(history)) + "\n", encoding="utf-8")


# @twin run_train
def run_train(preset: Preset, curve_path: Path | None = None) -> None:
    tokenizer = Tokenizer()
    data = load_dataset(preset, tokenizer)
    torch.manual_seed(preset.seed)
    model = GPTModel(preset.model)

    report.section("before training")
    report.scalar("val loss", evaluate(model, data.val, preset.train.batch_size))

    report.section("training")
    history = train_model(model, preset, data.train, data.val)
    for step, val_loss in zip(history.eval_steps, history.val_loss, strict=True):
        train_loss = history.train_loss[step - 1]
        report.text(f"step {step:04d}", f"train {train_loss:.4f}  val {val_loss:.4f}")
    report.count("optimizer steps", len(history.steps))
    report.count("tokens seen", history.tokens_seen)

    report.section("after training")
    report.scalar("val loss", evaluate(model, data.val, preset.train.batch_size))
    prompt = torch.tensor([tokenizer.encode("Every effort moves you")])
    out = generate_greedy(
        model, prompt, max_new_tokens=8, context_length=preset.model.context_length
    )
    report.show("generated ids", out)
    if curve_path is not None:
        save_history(history, curve_path)

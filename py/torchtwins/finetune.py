"""Lesson 07 - fine-tuning: turn the GPT into a spam classifier.

New head, frozen body, last-token classification, accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F  # noqa: N812

from torchtwins import report
from torchtwins.config import REPO_ROOT, Preset
from torchtwins.data import Windows, batches
from torchtwins.gpt import GPTModel
from torchtwins.modules import count_parameters
from torchtwins.tokenizer import Tokenizer

DEFAULT_SMS_PATH = REPO_ROOT / "assets" / "SMSSpamCollection.tsv"
LABELS = ("ham", "spam")


# @twin FinetuneHistory
@dataclass
class FinetuneHistory:
    # @step fields: per-epoch loss and accuracy
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


# @twin load_sms
def load_sms(path: Path = DEFAULT_SMS_PATH) -> list[tuple[int, str]]:
    # @step read: read lines
    rows: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line:
            # @step parse: label then tab then text
            # LABELS.index maps "ham"/"spam" -> 0/1 class ids.
            label, text = line.split("\t", 1)
            rows.append((LABELS.index(label), text))
    return rows


# @twin balanced
def balanced(rows: list[tuple[int, str]], per_class: int) -> list[tuple[int, str]]:
    """First ``per_class`` ham and spam messages, interleaved ham, spam, ham, spam, ..."""
    # @step take: first N of each class
    ham = [r for r in rows if r[0] == 0][:per_class]
    spam = [r for r in rows if r[0] == 1][:per_class]
    # @step interleave: ham, spam, ham, spam, ...
    # Keeps both classes in every mini-batch after the ordered 70/10/20 split.
    return [r for pair in zip(ham, spam, strict=True) for r in pair]


# @twin encode_examples
def encode_examples(
    examples: list[tuple[int, str]], tokenizer: Tokenizer, context_length: int
) -> Windows:
    """Truncate to ``context_length`` and *left*-pad, so the last position is always real."""
    # @step pad: fill with <|endoftext|>
    # inputs starts as [N, T] all pad ids.
    pad = tokenizer.end_of_text_id
    inputs = torch.full((len(examples), context_length), pad, dtype=torch.long)
    # @step fill: truncate and left-pad
    # Write tokens into the *right* edge so position T-1 is always a real token.
    # That is required for last-token classification below.
    for row, (_, text) in enumerate(examples):
        ids = tokenizer.encode(text)[:context_length]
        inputs[row, context_length - len(ids) :] = torch.tensor(ids)
    # @step labels: class-id vector
    # targets: [N] int64 (not [N, T] like LM windows).
    labels = torch.tensor([label for label, _ in examples])
    return Windows(inputs, labels)


# @twin split_examples
def split_examples(windows: Windows) -> tuple[Windows, Windows, Windows]:
    """70% train / 10% validation / 20% test, in order (the data is already interleaved)."""
    # @step cuts: 70 / 10 / 20 cut points
    n = len(windows)
    a, b = int(0.7 * n), int(0.8 * n)
    # @step parts: three Windows views
    parts = [(0, a), (a, b), (b, n)]
    train, val, test = (Windows(windows.inputs[s:e], windows.targets[s:e]) for s, e in parts)
    return train, val, test


# @twin freeze_body
def freeze_body(model: GPTModel, n_layers: int) -> None:
    """Train only the last Transformer block, the final LayerNorm and the new head."""
    # @step prefixes: trainable name prefixes
    trainable = (f"trf_blocks.{n_layers - 1}.", "final_norm.", "out_head.")
    # @step mark: toggle requires_grad by name
    # Freezing earlier blocks keeps most of the pretrained features; only the last
    # block + norm + 2-class head adapt to spam/ham.
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(trainable))


# @twin class_logits
def class_logits(model: GPTModel, inputs: torch.Tensor) -> torch.Tensor:
    # @step last: take the last time step
    # model(inputs): [B, T, C] with C=2 after replace_head.
    # [:, -1, :] -> [B, C]: classify from the final (rightmost) token state.
    logits: torch.Tensor = model(inputs)[:, -1, :]
    return logits


# @twin accuracy
def accuracy(model: GPTModel, windows: Windows, batch_size: int) -> float:
    # @step eval: eval mode
    model.eval()
    # @step count: count correct predictions
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in batches(windows, batch_size):
            correct += int((class_logits(model, inputs).argmax(-1) == labels).sum().item())
            total += labels.numel()
    # @step ratio: restore train mode and divide
    model.train()
    return correct / max(total, 1)


# @twin finetune_model
def finetune_model(
    model: GPTModel, preset: Preset, train: Windows, val: Windows
) -> FinetuneHistory:
    cfg = preset.finetune
    # @step params: only unfrozen tensors
    params = [p for p in model.parameters() if p.requires_grad]
    # @step optim: AdamW on that subset
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, foreach=False
    )
    # @step epoch: one pass over train
    generator = torch.Generator().manual_seed(preset.seed)
    history = FinetuneHistory()
    model.train()
    for _ in range(cfg.num_epochs):
        total, n = 0.0, 0
        for inputs, labels in batches(train, cfg.batch_size, generator):
            # CE on [B, 2] vs [B] — not the flattened LM layout.
            optimizer.zero_grad()
            loss = F.cross_entropy(class_logits(model, inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        # @step record: epoch metrics
        history.train_loss.append(total / max(n, 1))
        history.train_accuracy.append(accuracy(model, train, cfg.batch_size))
        history.val_accuracy.append(accuracy(model, val, cfg.batch_size))
    return history


# @twin classify
def classify(model: GPTModel, tokenizer: Tokenizer, text: str, context_length: int) -> str:
    # @step encode: encode one SMS
    # Reuses left-pad so the last position is still a real token.
    inputs = encode_examples([(0, text)], tokenizer, context_length).inputs
    # @step predict: argmax label
    model.eval()
    with torch.no_grad():
        label = int(class_logits(model, inputs).argmax(-1).item())
    return LABELS[label]


# @twin run_finetune
def run_finetune(preset: Preset) -> None:
    # @step data: load, balance, encode, split
    tokenizer = Tokenizer()
    cfg = preset.model
    examples = balanced(load_sms(), preset.finetune.examples_per_class)
    train, val, test = split_examples(encode_examples(examples, tokenizer, cfg.context_length))

    # @step report_data: dataset stats
    report.section("dataset")
    report.count("train examples", len(train))
    report.count("val examples", len(val))
    report.count("test examples", len(test))
    report.show("train labels", train.targets)

    # @step head: replace head and freeze body
    report.section("new head")
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)
    # out_head: Linear(emb_dim, vocab) -> Linear(emb_dim, 2).
    model.replace_head(len(LABELS))
    freeze_body(model, cfg.n_layers)
    report.count("parameters", count_parameters(model))
    report.count("trainable", sum(p.numel() for p in model.parameters() if p.requires_grad))
    report.scalar("test accuracy before", accuracy(model, test, preset.finetune.batch_size))

    # @step tune: fine-tune
    report.section("fine-tuning")
    history = finetune_model(model, preset, train, val)
    for epoch, (loss, train_acc, val_acc) in enumerate(
        zip(history.train_loss, history.train_accuracy, history.val_accuracy, strict=True), 1
    ):
        report.text(f"epoch {epoch}", f"loss {loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}")
    report.scalar("test accuracy after", accuracy(model, test, preset.finetune.batch_size))

    # @step classify: two held-out messages
    report.section("classify")
    for i in (-2, -1):  # the last two held-out test messages: one ham, one spam
        label, text = examples[i]
        predicted = classify(model, tokenizer, text, cfg.context_length)
        report.text(f"test[{i}]", f"true {LABELS[label]}  predicted {predicted}")

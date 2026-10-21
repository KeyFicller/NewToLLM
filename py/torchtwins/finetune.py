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
    # @step fields: 每个 epoch 的 loss 与准确率
    train_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


# @twin load_sms
def load_sms(path: Path = DEFAULT_SMS_PATH) -> list[tuple[int, str]]:
    # @step read: 逐行读文件
    rows: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line:
            # @step parse: 解析标签
            label, text = line.split("\t", 1)
            rows.append((LABELS.index(label), text))
    return rows


# @twin balanced
def balanced(rows: list[tuple[int, str]], per_class: int) -> list[tuple[int, str]]:
    """First ``per_class`` ham and spam messages, interleaved ham, spam, ham, spam, ..."""
    # @step take: 每类取前 N 条
    ham = [r for r in rows if r[0] == 0][:per_class]
    spam = [r for r in rows if r[0] == 1][:per_class]
    # @step interleave: 交错 ham/spam
    return [r for pair in zip(ham, spam, strict=True) for r in pair]


# @twin encode_examples
def encode_examples(
    examples: list[tuple[int, str]], tokenizer: Tokenizer, context_length: int
) -> Windows:
    """Truncate to ``context_length`` and *left*-pad, so the last position is always real."""
    # @step pad: 先全部填 <|endoftext|>
    pad = tokenizer.end_of_text_id
    inputs = torch.full((len(examples), context_length), pad, dtype=torch.long)
    # @step fill: 截断并左填充
    for row, (_, text) in enumerate(examples):
        ids = tokenizer.encode(text)[:context_length]
        inputs[row, context_length - len(ids) :] = torch.tensor(ids)
    # @step labels: 标签张量
    labels = torch.tensor([label for label, _ in examples])
    return Windows(inputs, labels)


# @twin split_examples
def split_examples(windows: Windows) -> tuple[Windows, Windows, Windows]:
    """70% train / 10% validation / 20% test, in order (the data is already interleaved)."""
    # @step cuts: 70/10/20 切分点
    n = len(windows)
    a, b = int(0.7 * n), int(0.8 * n)
    # @step parts: 切成三份
    parts = [(0, a), (a, b), (b, n)]
    train, val, test = (Windows(windows.inputs[s:e], windows.targets[s:e]) for s, e in parts)
    return train, val, test


# @twin freeze_body
def freeze_body(model: GPTModel, n_layers: int) -> None:
    """Train only the last Transformer block, the final LayerNorm and the new head."""
    # @step prefixes: 可训练的前缀
    trainable = (f"trf_blocks.{n_layers - 1}.", "final_norm.", "out_head.")
    # @step mark: 按名字开关梯度
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(trainable))


# @twin class_logits
def class_logits(model: GPTModel, inputs: torch.Tensor) -> torch.Tensor:
    # @step last: 取最后一个位置
    logits: torch.Tensor = model(inputs)[:, -1, :]
    return logits


# @twin accuracy
def accuracy(model: GPTModel, windows: Windows, batch_size: int) -> float:
    # @step eval: 切到 eval
    model.eval()
    # @step count: 统计预测正确数
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in batches(windows, batch_size):
            correct += int((class_logits(model, inputs).argmax(-1) == labels).sum().item())
            total += labels.numel()
    # @step ratio: 恢复训练并求比例
    model.train()
    return correct / max(total, 1)


# @twin finetune_model
def finetune_model(
    model: GPTModel, preset: Preset, train: Windows, val: Windows
) -> FinetuneHistory:
    cfg = preset.finetune
    # @step params: 只优化未冻结参数
    params = [p for p in model.parameters() if p.requires_grad]
    # @step optim: AdamW
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay, foreach=False
    )
    # @step epoch: 每个 epoch 训练一遍
    generator = torch.Generator().manual_seed(preset.seed)
    history = FinetuneHistory()
    model.train()
    for _ in range(cfg.num_epochs):
        total, n = 0.0, 0
        for inputs, labels in batches(train, cfg.batch_size, generator):
            optimizer.zero_grad()
            loss = F.cross_entropy(class_logits(model, inputs), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        # @step record: 记录 loss 与准确率
        history.train_loss.append(total / max(n, 1))
        history.train_accuracy.append(accuracy(model, train, cfg.batch_size))
        history.val_accuracy.append(accuracy(model, val, cfg.batch_size))
    return history


# @twin classify
def classify(model: GPTModel, tokenizer: Tokenizer, text: str, context_length: int) -> str:
    # @step encode: 编码单条短信
    inputs = encode_examples([(0, text)], tokenizer, context_length).inputs
    # @step predict: argmax 得标签
    model.eval()
    with torch.no_grad():
        label = int(class_logits(model, inputs).argmax(-1).item())
    return LABELS[label]


# @twin run_finetune
def run_finetune(preset: Preset) -> None:
    # @step data: 加载、平衡、编码、切分
    tokenizer = Tokenizer()
    cfg = preset.model
    examples = balanced(load_sms(), preset.finetune.examples_per_class)
    train, val, test = split_examples(encode_examples(examples, tokenizer, cfg.context_length))

    # @step report_data: 数据集统计
    report.section("dataset")
    report.count("train examples", len(train))
    report.count("val examples", len(val))
    report.count("test examples", len(test))
    report.show("train labels", train.targets)

    # @step head: 换头并冻结
    report.section("new head")
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)
    model.replace_head(len(LABELS))
    freeze_body(model, cfg.n_layers)
    report.count("parameters", count_parameters(model))
    report.count("trainable", sum(p.numel() for p in model.parameters() if p.requires_grad))
    report.scalar("test accuracy before", accuracy(model, test, preset.finetune.batch_size))

    # @step tune: 微调
    report.section("fine-tuning")
    history = finetune_model(model, preset, train, val)
    for epoch, (loss, train_acc, val_acc) in enumerate(
        zip(history.train_loss, history.train_accuracy, history.val_accuracy, strict=True), 1
    ):
        report.text(f"epoch {epoch}", f"loss {loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}")
    report.scalar("test accuracy after", accuracy(model, test, preset.finetune.batch_size))

    # @step classify: 分类两条留出短信
    report.section("classify")
    for i in (-2, -1):  # the last two held-out test messages: one ham, one spam
        label, text = examples[i]
        predicted = classify(model, tokenizer, text, cfg.context_length)
        report.text(f"test[{i}]", f"true {LABELS[label]}  predicted {predicted}")

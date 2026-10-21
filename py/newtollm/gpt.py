"""Lesson 05 - the GPT model: Transformer blocks, logits, greedy generation."""

from __future__ import annotations

import torch
from torch import nn

from newtollm import report
from newtollm.attention import MultiHeadAttention
from newtollm.config import GPTConfig, Preset
from newtollm.modules import FeedForward, LayerNorm, count_parameters
from newtollm.tokenizer import Tokenizer


# @twin TransformerBlock
class TransformerBlock(nn.Module):
    # @step layers: 注意力 + 前馈 + 两个 LayerNorm
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.drop_rate, cfg.n_heads, cfg.qkv_bias
        )
        self.ff = FeedForward(cfg.emb_dim)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    # @step residual: Pre-LN 残差
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        out: torch.Tensor = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return out


# @twin GPTModel
# @step blocks: 堆叠 N 个块
def make_blocks(cfg: GPTConfig) -> nn.Sequential:
    return nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])


# @step layers: 嵌入、块、输出头
class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = make_blocks(cfg)
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    # @step embed: 词嵌入 + 位置嵌入
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        tokens = idx.shape[1]
        positions = torch.arange(tokens, device=idx.device)
        x = self.drop_emb(self.tok_emb(idx) + self.pos_emb(positions))
        # @step stack: 过 N 个块 + LayerNorm + 输出头
        x = self.final_norm(self.trf_blocks(x))
        logits: torch.Tensor = self.out_head(x)
        return logits

    # @step head: 第 07 课用的换头
    def replace_head(self, num_outputs: int) -> None:
        """Swap the vocabulary projection for a fresh ``num_outputs``-way head (lesson 07)."""
        self.out_head = nn.Linear(self.out_head.in_features, num_outputs)


# @twin generate_greedy
def generate_greedy(
    model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_length: int
) -> torch.Tensor:
    # @step eval: 推理模式
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # @step crop: 截上下文、取最后一位
            logits = model(idx[:, -context_length:])[:, -1, :]
            # @step pick: 贪心选词并拼接
            next_id = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
    return idx


# @twin run_gpt
def run_gpt(preset: Preset) -> None:
    # @step build: 同种子建模型
    cfg = preset.model
    tokenizer = Tokenizer()
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)

    # @step logits: 前向出 logits
    report.section("GPTModel")
    report.count("parameters", count_parameters(model))
    prompt = torch.tensor([tokenizer.encode("Hello, I am")])
    report.show("prompt ids", prompt)
    model.eval()
    with torch.no_grad():
        report.show("logits", model(prompt))

    # @step generate: 生成并解码
    report.section("generate_greedy")
    out = generate_greedy(model, prompt, max_new_tokens=6, context_length=cfg.context_length)
    report.show("generated ids", out)
    report.text("decoded", tokenizer.decode(out[0].tolist()))

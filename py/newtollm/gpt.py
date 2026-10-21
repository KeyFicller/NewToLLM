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
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            cfg.emb_dim, cfg.emb_dim, cfg.context_length, cfg.drop_rate, cfg.n_heads, cfg.qkv_bias
        )
        self.ff = FeedForward(cfg.emb_dim)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_shortcut(self.att(self.norm1(x)))
        out: torch.Tensor = x + self.drop_shortcut(self.ff(self.norm2(x)))
        return out


# @twin GPTModel
class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        tokens = idx.shape[1]
        positions = torch.arange(tokens, device=idx.device)
        x = self.drop_emb(self.tok_emb(idx) + self.pos_emb(positions))
        x = self.final_norm(self.trf_blocks(x))
        logits: torch.Tensor = self.out_head(x)
        return logits


# @twin generate_greedy
def generate_greedy(
    model: GPTModel, idx: torch.Tensor, max_new_tokens: int, context_length: int
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx[:, -context_length:])[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
    return idx


# @twin run_gpt
def run_gpt(preset: Preset) -> None:
    cfg = preset.model
    tokenizer = Tokenizer()
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)

    report.section("GPTModel")
    report.count("parameters", count_parameters(model))
    prompt = torch.tensor([tokenizer.encode("Hello, I am")])
    report.show("prompt ids", prompt)
    model.eval()
    with torch.no_grad():
        report.show("logits", model(prompt))

    report.section("generate_greedy")
    out = generate_greedy(model, prompt, max_new_tokens=6, context_length=cfg.context_length)
    report.show("generated ids", out)
    report.text("decoded", tokenizer.decode(out[0].tolist()))

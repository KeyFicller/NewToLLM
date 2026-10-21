"""Lesson 04 - causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import nn

from torchtwins import report
from torchtwins.config import Preset


# @twin causal_mask
def causal_mask(context_length: int) -> torch.Tensor:
    """True above the diagonal: position t may not look at t+1, t+2, ..."""
    # @step triu: 上三角布尔掩码
    # Row t = query position, column s = key position. diagonal=1 marks every s > t,
    # i.e. the future. Those cells get -inf before softmax, and exp(-inf) = 0, so each
    # token's weights are spread only over itself and earlier tokens. That is what
    # lets one forward pass train every position at once without peeking ahead.
    return torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)


# @twin MultiHeadAttention
class MultiHeadAttention(nn.Module):
    mask: torch.Tensor

    # @step check: 检查 head 维度
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        drop_rate: float,
        n_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % n_heads != 0:
            msg = "d_out must be divisible by n_heads"
            raise ValueError(msg)
        # @step layers: Q/K/V 与输出投影
        self.d_out = d_out
        self.n_heads = n_heads
        self.head_dim = d_out // n_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(drop_rate)
        # A buffer, not a Parameter: it is module state that moves with .to(device) and
        # shows up in buffers(), but the optimizer never sees it. persistent=False keeps
        # it out of state_dict - it is rebuilt from context_length anyway.
        self.register_buffer("mask", causal_mask(context_length), persistent=False)

    # @step split: 拆成多头
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, tokens, d_out] -> [batch, heads, tokens, head_dim]"""
        batch, tokens, _ = x.shape
        # (batch, tokens, d_out) -> view (batch, tokens, heads, head_dim)
        # -> transpose (batch, heads, tokens, head_dim): heads become a batch-like dim,
        # so one matmul below attends in every head in parallel.
        return x.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

    # @step qk: 投影 Q/K 并打分
    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[1]
        queries = self.split_heads(self.w_query(x))
        keys = self.split_heads(self.w_key(x))
        # (b, h, T, hd) @ (b, h, hd, T) -> (b, h, T, T): score of every query vs every key.
        scores = queries @ keys.transpose(2, 3)
        # @step mask: 因果掩码
        # The buffer is sized for context_length; crop it to the actual sequence length.
        scores = scores.masked_fill(self.mask[:tokens, :tokens], float("-inf"))
        # @step softmax: 缩放 + softmax
        # Dividing by sqrt(head_dim) keeps the dot products' variance near 1, so softmax
        # does not saturate into a one-hot and gradients keep flowing. dim=-1 = over keys.
        return torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)

    # @step mix: dropout 后加权求和 V
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        weights = self.dropout(self.attention_weights(x))
        context = weights @ self.split_heads(self.w_value(x))
        # @step merge: 合并多头 + 输出投影
        # (b, h, T, hd) -> (b, T, h, hd) -> (b, T, d_out). transpose() leaves memory
        # non-contiguous, so contiguous() copies it before view() can merge the heads.
        context = context.transpose(1, 2).contiguous().view(batch, tokens, self.d_out)
        out: torch.Tensor = self.out_proj(context)
        return out


# @twin run_attention
def run_attention(preset: Preset) -> None:
    # @step input: 同种子输入
    cfg = preset.model
    torch.manual_seed(preset.seed)
    x = torch.randn(1, 4, cfg.emb_dim)

    # @step mask: 打印 mask
    report.section("causal_mask")
    report.show("causal_mask(4)", causal_mask(4))

    # @step attn: 注意力权重与输出
    report.section("MultiHeadAttention")
    attention = MultiHeadAttention(
        cfg.emb_dim, cfg.emb_dim, cfg.context_length, 0.0, cfg.n_heads, cfg.qkv_bias
    )
    weights = attention.attention_weights(x)
    report.show("weights[0, 0]", weights[0, 0])
    report.show("weights.sum(-1)", weights.sum(dim=-1))
    report.show("attention(x)", attention(x))

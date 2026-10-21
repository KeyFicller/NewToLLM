"""Lesson 03 - modules: GELU, LayerNorm and the Transformer feed-forward block."""

from __future__ import annotations

import math

import torch
from torch import nn

from torchtwins import report
from torchtwins.config import Preset


# @twin GELU
class GELU(nn.Module):
    """tanh approximation used by GPT-2."""

    # @step formula: tanh 近似公式
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


# @twin LayerNorm
class LayerNorm(nn.Module):
    # @step params: 注册 scale / shift
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # nn.Parameter = a tensor the module owns AND the optimizer should train.
        # Assigning it as an attribute registers it automatically.
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # @step stats: 均值与方差
        # Normalise over the embedding dim only: (batch, tokens, emb) -> (batch, tokens, 1).
        # keepdim=True keeps the trailing 1 so the result broadcasts back against x.
        mean = x.mean(dim=-1, keepdim=True)
        # unbiased=False divides by N, not N-1 - this matches GPT-2's LayerNorm.
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # @step normalize: 标准化再仿射
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


# @twin FeedForward
class FeedForward(nn.Module):
    # @step layers: Linear → GELU → Linear
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # Expand to 4x width, apply the non-linearity there, project back:
            # (batch, tokens, emb) -> (batch, tokens, 4*emb) -> (batch, tokens, emb).
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    # @step forward: 前向
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out


# @twin count_parameters
def count_parameters(module: nn.Module) -> int:
    # @step sum: 参数计数
    return sum(p.numel() for p in module.parameters())


# @twin run_modules
def run_modules(preset: Preset) -> None:
    # @step input: 同种子输入
    emb_dim = preset.model.emb_dim
    torch.manual_seed(preset.seed)
    x = torch.randn(2, 3, emb_dim)

    # @step gelu: GELU
    report.section("GELU")
    report.show("gelu(linspace(-3, 3, 7))", GELU()(torch.linspace(-3.0, 3.0, 7)))

    # @step layernorm: LayerNorm
    report.section("LayerNorm")
    normed = LayerNorm(emb_dim)(x)
    report.show("mean(-1)", normed.mean(dim=-1))
    report.show("var(-1)", normed.var(dim=-1, unbiased=False))

    # @step ff: FeedForward
    report.section("FeedForward")
    ff = FeedForward(emb_dim)
    report.count("parameters", count_parameters(ff))
    report.show("ff(x)", ff(x))

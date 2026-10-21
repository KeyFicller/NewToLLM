"""Lesson 03 - modules: GELU, LayerNorm and the Transformer feed-forward block."""

from __future__ import annotations

import math

import torch
from torch import nn

from newtollm import report
from newtollm.config import Preset


# @twin GELU
class GELU(nn.Module):
    """tanh approximation used by GPT-2."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


# @twin LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


# @twin FeedForward
class FeedForward(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layers(x)
        return out


# @twin count_parameters
def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# @twin run_modules
def run_modules(preset: Preset) -> None:
    emb_dim = preset.model.emb_dim
    torch.manual_seed(preset.seed)
    x = torch.randn(2, 3, emb_dim)

    report.section("GELU")
    report.show("gelu(linspace(-3, 3, 7))", GELU()(torch.linspace(-3.0, 3.0, 7)))

    report.section("LayerNorm")
    normed = LayerNorm(emb_dim)(x)
    report.show("mean(-1)", normed.mean(dim=-1))
    report.show("var(-1)", normed.var(dim=-1, unbiased=False))

    report.section("FeedForward")
    ff = FeedForward(emb_dim)
    report.count("parameters", count_parameters(ff))
    report.show("ff(x)", ff(x))

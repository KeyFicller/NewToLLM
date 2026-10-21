"""Lesson 02 - autograd: gradients, no_grad, and a hand-written SGD loop."""

from __future__ import annotations

import torch

from newtollm import report
from newtollm.config import Preset


# @twin gradients
def gradients() -> None:
    report.section("gradients")
    x = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 4.0, 6.0])
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)
    loss = ((w * x + b - target) ** 2).mean()
    loss.backward()
    report.scalar("loss", loss.item())
    report.show("w.grad", w.grad if w.grad is not None else torch.empty(0))
    report.show("b.grad", b.grad if b.grad is not None else torch.empty(0))


# @twin no_grad
def no_grad() -> None:
    report.section("no_grad")
    w = torch.ones(3, requires_grad=True)
    with torch.no_grad():
        frozen = w * 2
    tracked = w * 2
    report.text("frozen.requires_grad", str(frozen.requires_grad).lower())
    report.text("tracked.requires_grad", str(tracked.requires_grad).lower())


# @twin fit_line
def fit_line(seed: int, steps: int = 50, lr: float = 0.1) -> None:
    report.section("fit_line")
    torch.manual_seed(seed)
    x = torch.randn(32, 1)
    y = 3.0 * x + 0.5 + 0.1 * torch.randn(32, 1)
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        loss = ((x * w + b - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            assert w.grad is not None
            assert b.grad is not None
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    report.show("w", w)
    report.show("b", b)
    report.scalar("final loss", loss.item())


# @twin run_autograd
def run_autograd(preset: Preset) -> None:
    gradients()
    no_grad()
    fit_line(preset.seed)

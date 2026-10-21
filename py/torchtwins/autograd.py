"""Lesson 02 - autograd: gradients, no_grad, and a hand-written SGD loop."""

from __future__ import annotations

import torch

from torchtwins import report
from torchtwins.config import Preset


# @twin gradients
def gradients() -> None:
    report.section("gradients")
    # @step data: 输入与目标
    x = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 4.0, 6.0])
    # @step leaf: 可求导的叶子
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.1, requires_grad=True)
    # @step loss: 均方误差
    loss = ((w * x + b - target) ** 2).mean()
    # @step backward: 反向传播
    loss.backward()
    # @step read: 读出梯度
    report.scalar("loss", loss.item())
    report.show("w.grad", w.grad if w.grad is not None else torch.empty(0))
    report.show("b.grad", b.grad if b.grad is not None else torch.empty(0))


# @twin no_grad
def no_grad() -> None:
    report.section("no_grad")
    # @step leaf: 需要梯度的张量
    w = torch.ones(3, requires_grad=True)
    # @step guard: 关闭建图
    with torch.no_grad():
        frozen = w * 2
    # @step tracked: 守卫之外照常建图
    tracked = w * 2
    # @step report: 打印布尔
    report.text("frozen.requires_grad", str(frozen.requires_grad).lower())
    report.text("tracked.requires_grad", str(tracked.requires_grad).lower())


# @twin fit_line
def fit_line(seed: int, steps: int = 50, lr: float = 0.1) -> None:
    report.section("fit_line")
    # @step data: 造数据 y=3x+0.5
    torch.manual_seed(seed)
    x = torch.randn(32, 1)
    y = 3.0 * x + 0.5 + 0.1 * torch.randn(32, 1)
    # @step params: 参数清零
    w = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    loss = torch.tensor(0.0)
    for _ in range(steps):
        # @step forward: 前向 + 反向
        loss = ((x * w + b - y) ** 2).mean()
        loss.backward()
        # @step sgd: 手写 SGD 更新
        with torch.no_grad():
            assert w.grad is not None
            assert b.grad is not None
            w -= lr * w.grad
            b -= lr * b.grad
            w.grad.zero_()
            b.grad.zero_()
    # @step report: 打印结果
    report.show("w", w)
    report.show("b", b)
    report.scalar("final loss", loss.item())


# @twin run_autograd
def run_autograd(preset: Preset) -> None:
    # @step pipeline: 按顺序调用
    gradients()
    no_grad()
    fit_line(preset.seed)

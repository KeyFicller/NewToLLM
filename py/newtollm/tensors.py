"""Lesson 01 - tensors: creation, shapes, indexing, broadcasting, matmul."""

from __future__ import annotations

import torch

from newtollm import report
from newtollm.config import Preset


# @twin create
def create(seed: int) -> torch.Tensor:
    report.section("create")
    # @step make: 造张量
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    report.show("arange.reshape(2, 3)", x)
    report.show("zeros(2, 2)", torch.zeros(2, 2))
    report.show("full((2, 2), 7)", torch.full((2, 2), 7.0))
    # @step seeded: 同种子随机数
    torch.manual_seed(seed)
    report.show("randn(2, 3)", torch.randn(2, 3))
    # @step literal: 嵌套字面量
    report.show("tensor([[1, 2], [3, 4]])", torch.tensor([[1, 2], [3, 4]]))
    return x


# @twin reshape
def reshape(x: torch.Tensor) -> None:
    report.section("reshape")
    # @step view: 改形状
    report.show("x.view(3, 2)", x.view(3, 2))
    report.show("x.transpose(0, 1)", x.transpose(0, 1))
    report.show("x.unsqueeze(0)", x.unsqueeze(0))
    report.show("x.flatten()", x.flatten())
    # @step size: 取维度
    report.count("x.size(-1)", x.size(-1))


# @twin index
def index(x: torch.Tensor) -> None:
    report.section("index")
    # @step int: 整数下标
    report.show("x[0]", x[0])
    # @step slice: 切片
    report.show("x[:, 1:]", x[:, 1:])
    report.show("x[:, -1]", x[:, -1])
    # @step mask: 布尔掩码
    report.show("x > 2", x > 2)
    report.show("x[x > 2]", x[x > 2])


# @twin broadcast
def broadcast(x: torch.Tensor) -> None:
    report.section("broadcast")
    # @step bias: 一维偏置
    bias = torch.tensor([10.0, 20.0, 30.0])
    # @step ops: 广播运算
    report.show("x + bias", x + bias)
    report.show("x * 2", x * 2)
    # @step reduce: 归约
    report.show("x.mean(dim=-1, keepdim=True)", x.mean(dim=-1, keepdim=True))


# @twin matmul
def matmul(x: torch.Tensor) -> None:
    report.section("matmul")
    # @step matmul: 矩阵乘
    report.show("x @ x.T", x @ x.T)
    # @step softmax: softmax
    report.show("softmax(x, dim=-1)", torch.softmax(x, dim=-1))


# @twin run_tensors
def run_tensors(preset: Preset) -> None:
    # @step pipeline: 按课时顺序调用
    x = create(preset.seed)
    reshape(x)
    index(x)
    broadcast(x)
    matmul(x)

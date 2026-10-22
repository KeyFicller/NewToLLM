"""Deterministic, language-neutral printing.

Both sides print through ``report`` so their stdout can be diffed line by line.
"""

from __future__ import annotations

import torch

MAX_VALUES = 8


# @twin format_value
def format_value(value: float | int | bool, dtype: torch.dtype) -> str:
    # @step bool: 布尔打成小写 true/false
    if dtype == torch.bool:
        return "true" if value else "false"
    # @step float: 浮点固定四位小数
    if dtype.is_floating_point:
        return f"{float(value):.4f}"
    # @step int: 其余按整数输出
    return str(int(value))


# @twin section
def section(title: str) -> None:
    # @step banner: 课程小节标题行
    print(f"== {title} ==")


# @twin show
def show(label: str, tensor: torch.Tensor) -> None:
    # @step flatten: detach 后拉平并拷到 CPU
    flat = tensor.detach().flatten().cpu()
    # @step head: 最多打印前 MAX_VALUES 个 多的用 ...
    head = [format_value(v, flat.dtype) for v in flat[:MAX_VALUES].tolist()]
    if flat.numel() > MAX_VALUES:
        head.append("...")
    # @step sum: bool 用 int64 求和 其余保持原 dtype
    total = flat.sum() if flat.dtype != torch.bool else flat.sum(dtype=torch.int64)
    sum_dtype = torch.int64 if flat.dtype == torch.bool else flat.dtype
    # @step print: 统一 label / shape / values / sum 一行
    shape = ", ".join(str(d) for d in tensor.shape)
    print(
        f"{label}: shape=[{shape}] values=[{', '.join(head)}] "
        f"sum={format_value(total.item(), sum_dtype)}"
    )


# @twin scalar
def scalar(label: str, value: float) -> None:
    # @step print: 标量也固定四位小数
    print(f"{label}: {value:.4f}")


# @twin count
def count(label: str, value: int) -> None:
    # @step print: 整数计数无小数
    print(f"{label}: {value}")


# @twin text
def text(label: str, value: str) -> None:
    # @step print: 任意字符串标签行
    print(f"{label}: {value}")

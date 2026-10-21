"""Deterministic, language-neutral printing.

Both sides print through ``report`` so their stdout can be diffed line by line.
"""

from __future__ import annotations

import torch

MAX_VALUES = 8


# @twin format_value
def format_value(value: float | int | bool, dtype: torch.dtype) -> str:
    if dtype == torch.bool:
        return "true" if value else "false"
    if dtype.is_floating_point:
        return f"{float(value):.4f}"
    return str(int(value))


# @twin section
def section(title: str) -> None:
    print(f"== {title} ==")


# @twin show
def show(label: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().flatten().cpu()
    head = [format_value(v, flat.dtype) for v in flat[:MAX_VALUES].tolist()]
    if flat.numel() > MAX_VALUES:
        head.append("...")
    total = flat.sum() if flat.dtype != torch.bool else flat.sum(dtype=torch.int64)
    sum_dtype = torch.int64 if flat.dtype == torch.bool else flat.dtype
    shape = ", ".join(str(d) for d in tensor.shape)
    print(
        f"{label}: shape=[{shape}] values=[{', '.join(head)}] "
        f"sum={format_value(total.item(), sum_dtype)}"
    )


# @twin scalar
def scalar(label: str, value: float) -> None:
    print(f"{label}: {value:.4f}")


# @twin count
def count(label: str, value: int) -> None:
    print(f"{label}: {value}")


# @twin text
def text(label: str, value: str) -> None:
    print(f"{label}: {value}")

"""Deterministic, language-neutral printing.

Both sides print through ``report`` so their stdout can be diffed line by line.
"""

from __future__ import annotations

import torch

MAX_VALUES = 8


# @twin format_value
def format_value(value: float | int | bool, dtype: torch.dtype) -> str:
    # @step bool: print bools as lowercase true/false
    if dtype == torch.bool:
        return "true" if value else "false"
    # @step float: floats fixed to four decimal places
    if dtype.is_floating_point:
        return f"{float(value):.4f}"
    # @step int: everything else as an integer
    return str(int(value))


# @twin section
def section(title: str) -> None:
    # @step banner: lesson section banner line
    print(f"== {title} ==")


# @twin show
def show(label: str, tensor: torch.Tensor) -> None:
    # @step flatten: detach, flatten, copy to CPU so printing ignores grads/device
    flat = tensor.detach().flatten().cpu()
    # @step head: print at most MAX_VALUES values, then ...
    head = [format_value(v, flat.dtype) for v in flat[:MAX_VALUES].tolist()]
    if flat.numel() > MAX_VALUES:
        head.append("...")
    # @step sum: sum bools as int64; keep other dtypes as-is
    total = flat.sum() if flat.dtype != torch.bool else flat.sum(dtype=torch.int64)
    sum_dtype = torch.int64 if flat.dtype == torch.bool else flat.dtype
    # @step print: one line with label / shape / values / sum
    shape = ", ".join(str(d) for d in tensor.shape)
    print(
        f"{label}: shape=[{shape}] values=[{', '.join(head)}] "
        f"sum={format_value(total.item(), sum_dtype)}"
    )


# @twin scalar
def scalar(label: str, value: float) -> None:
    # @step print: scalars also fixed to four decimals
    print(f"{label}: {value:.4f}")


# @twin count
def count(label: str, value: int) -> None:
    # @step print: integer counts, no decimals
    print(f"{label}: {value}")


# @twin text
def text(label: str, value: str) -> None:
    # @step print: free-form string label line
    print(f"{label}: {value}")

"""CLI - mirrors ``cpp/newtollm/main.cpp``.

python -m newtollm <lesson> [--preset tiny|small|gpt2] [--seed N]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from newtollm.attention import run_attention
from newtollm.autograd import run_autograd
from newtollm.config import Preset, load_preset
from newtollm.gpt import run_gpt
from newtollm.modules import run_modules
from newtollm.tensors import run_tensors
from newtollm.tokenizer import Tokenizer

LESSONS: dict[str, tuple[str, Callable[[Preset], None]]] = {
    "tensors": ("01 creation, shapes, indexing, broadcasting", run_tensors),
    "autograd": ("02 gradients, no_grad, hand-written SGD", run_autograd),
    "modules": ("03 GELU, LayerNorm, FeedForward", run_modules),
    "attention": ("04 causal multi-head self-attention", run_attention),
    "gpt": ("05 GPT model and greedy generation", run_gpt),
}


def tokenize_file(path: Path) -> None:
    ids = Tokenizer().encode(path.read_text(encoding="utf-8"))
    print(len(ids))
    print(" ".join(str(i) for i in ids))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="newtollm", description=__doc__)
    parser.add_argument("lesson", choices=[*LESSONS, "list", "tokenize"])
    parser.add_argument("path", nargs="?", type=Path, help="file for 'tokenize'")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.lesson == "list":
        for name, (description, _) in LESSONS.items():
            print(f"{name:<10} {description}")
        return 0
    if args.lesson == "tokenize":
        if args.path is None:
            parser.error("tokenize needs a file path")
        tokenize_file(args.path)
        return 0

    preset = load_preset(args.preset)
    if args.seed is not None:
        preset = preset.with_seed(args.seed)
    LESSONS[args.lesson][1](preset)
    return 0


if __name__ == "__main__":
    sys.exit(main())

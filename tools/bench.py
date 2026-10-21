"""Benchmark: the same GPT workloads timed in PyTorch (in-process) and LibTorch.

    python tools/bench.py [--presets tiny small] [--threads 4] [--warmup 5] [--iters 30] [--write]

Both halves use the same preset, seed, input shape, thread count and timing method
(warm-up, then the median of ``--iters`` wall-clock samples). ``--write`` stores the
raw samples and medians in docs/results/benchmark.json; the site only plots that file.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

import torch

from newtollm.config import REPO_ROOT, load_preset
from newtollm.gpt import GPTModel
from newtollm.train import loss_batch

BENCH_EXE = REPO_ROOT / "build" / "bin" / "bench"
RESULTS = REPO_ROOT / "docs" / "results" / "benchmark.json"


def time_ms(fn: Callable[[], None], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def bench_python(preset_name: str, threads: int, warmup: int, iters: int) -> dict[str, Any]:
    torch.set_num_threads(threads)
    preset = load_preset(preset_name)
    cfg = preset.model
    torch.manual_seed(preset.seed)
    model = GPTModel(cfg)
    shape = (preset.train.batch_size, cfg.context_length)
    inputs = torch.randint(cfg.vocab_size, shape)
    targets = torch.randint(cfg.vocab_size, shape)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=preset.train.learning_rate,
        weight_decay=preset.train.weight_decay,
        foreach=False,
    )

    def forward() -> None:
        with torch.no_grad():
            model(inputs)

    def train_step() -> None:
        optimizer.zero_grad()
        loss_batch(model, inputs, targets).backward()
        optimizer.step()

    model.eval()
    fwd = time_ms(forward, warmup, iters)
    model.train()
    step = time_ms(train_step, warmup, iters)
    return {
        "forward_ms": statistics.median(fwd),
        "train_step_ms": statistics.median(step),
        "forward_samples": fwd,
        "train_step_samples": step,
    }


def bench_cpp(preset_name: str, threads: int, warmup: int, iters: int) -> dict[str, Any]:
    out = subprocess.run(
        [str(BENCH_EXE), preset_name, str(threads), str(warmup), str(iters)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result: dict[str, Any] = json.loads(out.strip().splitlines()[-1])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=["tiny", "small"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--write", action="store_true", help="write docs/results/benchmark.json")
    args = parser.parse_args(argv)
    if not BENCH_EXE.exists():
        sys.exit(f"[bench] {BENCH_EXE} not found - build the C++ side first")

    rows: list[dict[str, Any]] = []
    for name in args.presets:
        cfg = load_preset(name)
        row = {
            "preset": name,
            "batch": [cfg.train.batch_size, cfg.model.context_length],
            "python": bench_python(name, args.threads, args.warmup, args.iters),
            "cpp": bench_cpp(name, args.threads, args.warmup, args.iters),
        }
        rows.append(row)
        for metric in ("forward_ms", "train_step_ms"):
            py, cpp = row["python"][metric], row["cpp"][metric]
            print(
                f"[bench] {name:<6} {metric:<14} python={py:8.2f}  cpp={cpp:8.2f}  "
                f"ratio={cpp / py:.2f}"
            )
    if args.write:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pytorch": torch.__version__,
            "libtorch": subprocess.run(
                [str(BENCH_EXE), "version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "machine": f"{platform.system()} {platform.machine()}",
            "threads": args.threads,
            "warmup": args.warmup,
            "iters": args.iters,
            "results": rows,
        }
        RESULTS.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Training-curve parity gate: PyTorch and LibTorch must train *the same way*.

Both CLIs run ``train --curve`` with the same preset and seed; the gate requires the
same number of optimizer steps (the old project did 85 vs 75) and every per-step
loss within ``ATOL``.

    python tools/train_parity.py [--preset tiny] [--seed 123] [--write]

``--write`` stores both curves in docs/results/train_curves.json for the site.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from newtollm.config import REPO_ROOT

CPP_CLI = REPO_ROOT / "build" / "bin" / "newtollm"
RESULTS = REPO_ROOT / "docs" / "results" / "train_curves.json"
ATOL = 1e-4


@dataclass(frozen=True)
class TrainParity:
    preset: str
    seed: int
    python_steps: int
    cpp_steps: int
    max_train_diff: float
    max_val_diff: float
    passed: bool


def run_curve(command: list[str], preset: str, seed: int, out: Path) -> dict[str, Any]:
    subprocess.run(
        [*command, "train", "--preset", preset, "--seed", str(seed), "--curve", str(out)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    curve: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return curve


def max_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0)


def compare(preset: str, seed: int) -> tuple[TrainParity, dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        py = run_curve([sys.executable, "-m", "newtollm"], preset, seed, tmp / "py.json")
        cpp = run_curve([str(CPP_CLI)], preset, seed, tmp / "cpp.json")
    train_diff = max_diff(py["train_loss"], cpp["train_loss"])
    val_diff = max_diff(py["val_loss"], cpp["val_loss"])
    result = TrainParity(
        preset=preset,
        seed=seed,
        python_steps=len(py["steps"]),
        cpp_steps=len(cpp["steps"]),
        max_train_diff=train_diff,
        max_val_diff=val_diff,
        passed=len(py["steps"]) == len(cpp["steps"]) and max(train_diff, val_diff) <= ATOL,
    )
    return result, py, cpp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--write", action="store_true", help="write docs/results/train_curves.json")
    args = parser.parse_args(argv)
    if not CPP_CLI.exists():
        sys.exit(f"[train-parity] {CPP_CLI} not found - build the C++ side first")

    result, py, cpp = compare(args.preset, args.seed)
    print(
        f"[train-parity] {result.preset} seed={result.seed} steps py={result.python_steps} "
        f"cpp={result.cpp_steps} max|train diff|={result.max_train_diff:.3e} "
        f"max|val diff|={result.max_val_diff:.3e}  {'PASS' if result.passed else 'FAIL'}"
    )
    if args.write:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        payload = {"atol": ATOL, "result": asdict(result), "python": py, "cpp": cpp}
        RESULTS.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())

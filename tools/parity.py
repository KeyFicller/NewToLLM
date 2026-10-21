"""Numerical parity gate: the PyTorch and LibTorch GPTModel must agree.

Two independent checks per preset:

* ``init``    - same seed => C++ draws the *same* random weights as Python
                (member order, argument evaluation order and RNG usage all match).
* ``forward`` - Python's weights copied into the C++ model => same logits.

    python tools/parity.py [--presets tiny small gpt2] [--write]

``--write`` stores the measured numbers in docs/results/parity.json for the site.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from newtollm.config import REPO_ROOT, load_preset
from newtollm.gpt import GPTModel

PARITY_EXE = REPO_ROOT / "build" / "bin" / "parity"
RESULTS = REPO_ROOT / "docs" / "results" / "parity.json"
ATOL = 1e-4


@dataclass(frozen=True)
class ParityResult:
    preset: str
    check: str
    tensors: int
    max_abs_diff: float
    passed: bool


def load_trusted(path: Path) -> list[torch.Tensor]:
    # The file was written seconds ago by our own C++ binary; torch::pickle_save emits
    # TorchScript type tags that the weights_only unpickler does not allow-list.
    tensors: list[torch.Tensor] = torch.load(path, weights_only=False)
    return tensors


def seeded_model(preset_name: str, seed: int) -> GPTModel:
    preset = load_preset(preset_name)
    torch.manual_seed(seed)
    return GPTModel(preset.model).eval()


def check_init(preset_name: str, seed: int, tmp: Path) -> ParityResult:
    out = tmp / "init.pt"
    names = subprocess.run(
        [str(PARITY_EXE), "init", preset_name, str(seed), str(out)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    py_params = dict(seeded_model(preset_name, seed).named_parameters())
    if names != list(py_params):
        msg = f"parameter names differ:\n{names}\n{list(py_params)}"
        raise AssertionError(msg)
    diff = max(
        (p.detach() - c).abs().max().item()
        for p, c in zip(py_params.values(), load_trusted(out), strict=True)
    )
    return ParityResult(preset_name, "init", len(names), diff, diff == 0.0)


def check_forward(preset_name: str, seed: int, tmp: Path) -> ParityResult:
    model = seeded_model(preset_name, seed + 1)
    cfg = load_preset(preset_name).model
    torch.manual_seed(seed)
    idx = torch.randint(0, cfg.vocab_size, (2, min(16, cfg.context_length)))
    with torch.no_grad():
        py_logits = model(idx)
    weights = [p.detach().contiguous() for p in model.parameters()]
    torch.save((idx, weights), tmp / "in.pt")
    subprocess.run(
        [str(PARITY_EXE), "forward", preset_name, str(tmp / "in.pt"), str(tmp / "out.pt")],
        check=True,
    )
    (cpp_logits,) = load_trusted(tmp / "out.pt")
    diff = (py_logits - cpp_logits).abs().max().item()
    same_argmax = torch.equal(py_logits.argmax(-1), cpp_logits.argmax(-1))
    return ParityResult(preset_name, "forward", len(weights), diff, diff <= ATOL and same_argmax)


def run(presets: list[str], seed: int) -> list[ParityResult]:
    results: list[ParityResult] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        for name in presets:
            results.append(check_init(name, seed, tmp))
            results.append(check_forward(name, seed, tmp))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=["tiny", "small"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--write", action="store_true", help="write docs/results/parity.json")
    args = parser.parse_args(argv)
    if not PARITY_EXE.exists():
        sys.exit(f"[parity] {PARITY_EXE} not found - build the C++ side first")

    results = run(args.presets, args.seed)
    for r in results:
        verdict = "PASS" if r.passed else "FAIL"
        print(
            f"[parity] {r.preset:<6} {r.check:<8} tensors={r.tensors:<4} "
            f"max|diff|={r.max_abs_diff:.3e}  {verdict}"
        )
    if args.write:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pytorch": torch.__version__,
            "libtorch": subprocess.run(
                [str(PARITY_EXE), "version"], check=True, capture_output=True, text=True
            ).stdout.strip(),
            "machine": f"{platform.system()} {platform.machine()}",
            "atol": ATOL,
            "results": [asdict(r) for r in results],
        }
        RESULTS.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Numerical parity check: PyTorch ToyModel vs LibTorch ToyModel.

1. Python builds the toy model (seeded), runs a forward pass on a fixed input.
2. Weights + input are handed to the C++ `parity_check` executable, which loads
   them into the C++ ToyModel and runs the same forward pass.
3. Logits are compared; exits non-zero if they differ by more than --atol.

    python tools/parity_check.py               # small model (fast)
    python tools/parity_check.py --full        # GPT-2 124M shape
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from python_impl.toy_model.model import ToyModel  # noqa: E402
from python_impl.utils import runtime  # noqa: E402

DEFAULT_EXE = REPO_ROOT / "build" / "cpp_impl" / "tools" / "parity_check"


def run_parity(full=False, seed=None, atol=1e-4, exe=DEFAULT_EXE, batch=2, n_tokens=16):
    runtime.set_options(seed=seed, small=not full)
    cfg = runtime.model_config()
    torch.manual_seed(runtime.seed())
    model = ToyModel(cfg).eval()
    inp = torch.randint(0, cfg["vocab_size"], (batch, n_tokens), dtype=torch.long)
    with torch.no_grad():
        py_logits = model(inp)

    with tempfile.TemporaryDirectory() as tmp:
        in_path, out_path = Path(tmp) / "in.pt", Path(tmp) / "out.pt"
        state = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
        torch.save({"config": cfg, "state": state, "input": inp}, in_path)
        subprocess.run([str(exe), str(in_path), str(out_path)], check=True)
        cpp_logits = torch.load(out_path)

    max_abs = (py_logits - cpp_logits).abs().max().item()
    same_argmax = torch.equal(py_logits.argmax(-1), cpp_logits.argmax(-1))
    ok = max_abs <= atol and same_argmax
    print(
        f"[parity] model={'full' if full else 'small'} logits{tuple(py_logits.shape)} "
        f"max_abs_diff={max_abs:.3e} (atol={atol:g}) argmax_equal={same_argmax} -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok, max_abs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="use the full GPT-2 124M shape instead of model_small")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="path to the C++ parity_check executable")
    args = ap.parse_args()
    if not args.exe.exists():
        sys.exit(f"[parity] C++ executable not found: {args.exe} (build with cmake first)")
    ok, _ = run_parity(full=args.full, seed=args.seed, atol=args.atol, exe=args.exe)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

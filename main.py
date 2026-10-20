"""Python (PyTorch) entrypoint.

Task names are kept identical to the C++ (LibTorch) entrypoint `sandbox`,
so the same experiment can be run on both sides:

    python main.py <task>
    ./build/cpp_impl/sandbox/sandbox <task>

Common flags (same on both sides): --seed N, --device cpu|cuda|mps, --small.
"""

import argparse
import sys

# Ordered task table: name -> (description, module path, function name).
# Modules are imported lazily so that e.g. `verify` does not require
# optional dependencies (transformers / pandas) used by other tasks.
TASKS = {
    "verify": ("torch version / device check", "python_impl.verifier.torch_verifier", "verify_torch"),
    "brief": ("torch basic examples", "python_impl.basic.torch_brief", "brief_torch"),
    "toy_model": ("toy model generation example", "python_impl.toy_model.torch_toy_model", "toy_model_torch"),
    "train": ("toy model pre-training example", "python_impl.train.torch_train", "train_torch"),
    "load_model": ("load public GPT-2 weights into toy model", "python_impl.load_model.torch_load_model", "load_model_torch"),
    "fine_tuning": ("fine-tuning example", "python_impl.fine_tuning.torch_fine_tuning", "fine_tuning_torch"),
}


def run_task(name):
    import importlib

    desc, module_name, func_name = TASKS[name]
    print(f"--- [{name}] {desc} ---")
    getattr(importlib.import_module(module_name), func_name)()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LLM_from_scratch (PyTorch) entrypoint")
    parser.add_argument(
        "task",
        nargs="?",
        default="verify",
        choices=[*TASKS, "all", "list"],
        help="Task to run (default: verify; 'all' runs every task in order; 'list' prints tasks)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: configs/toy_gpt2.json)")
    parser.add_argument("--device", default=None, help="force device, e.g. cpu / cuda / mps (default: auto)")
    parser.add_argument("--small", action="store_true", help="use the small toy model (model_small) for toy_model / train")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from python_impl.utils import runtime

    runtime.set_options(seed=args.seed, device=args.device, small=args.small)
    if args.task == "list":
        for name, (desc, _, _) in TASKS.items():
            print(f"{name:<12} {desc}")
        return 0
    names = list(TASKS) if args.task == "all" else [args.task]
    for name in names:
        run_task(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

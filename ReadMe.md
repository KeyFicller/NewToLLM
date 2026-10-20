## LLM from Scratch

This project explores LLM-related implementation from scratch using Torch in both:

- Python (`PyTorch`)
- C++ (`LibTorch`)

The two sides are meant to be compared side by side: every task has the same
name in both entrypoints. See [docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md) for
the ongoing refactor plan.

## Project Structure

- `python_impl/`: Python-side implementation and verification code
- `cpp_impl/`: C++-side implementation and verification code
- `extras/`: Python-only extras without a C++ counterpart (see `extras/README.md`)
- `configs/`: shared model / training config used by both sides
- `tools/`: cross-side checks (`parity_check.py`, C++ half in `cpp_impl/tools/`)
- `assets/`: datasets and the GPT-2 tokenizer file shared by both sides
- `main.py`: Python entrypoint
- `cpp_impl/sandbox/main.cpp`: C++ entrypoint (`sandbox` executable)

## Python ↔ C++ layout

| topic | Python (`python_impl/`) | C++ (`cpp_impl/my_torch/`) |
|-------|-------------------------|----------------------------|
| verify | `verifier/torch_verifier.py` | `verifier/torch_verifier.cpp` |
| basics | `basic/torch_brief.py` | `basic/torch_brief.cpp` |
| model | `toy_model/{config,transformer,model,torch_toy_model}.py` | `toy_model/{config.h,transformer,model,torch_toy_model}.cpp` |
| pre-train | `train/{data_utils,torch_train}.py` | `train/{data_utils,torch_train}.cpp` |
| load GPT-2 | `load_model/torch_load_model.py` | `load_model/torch_load_model.cpp` |
| fine-tune | `fine_tuning/{fine_tuning_classify,fine_tuning_instruction,utils}.py` | `fine_tuning/{torch_fine_tuning,fine_tuning_classify}.cpp` (instruction: ❌ not yet) |
| utils | `utils/torch_utils.py` | `utils/torch_utils.cpp` |

## Tasks

| task          | what it does                                  |
|---------------|-----------------------------------------------|
| `verify`      | print torch version and selected device (default) |
| `brief`       | torch basic examples                          |
| `toy_model`   | build the toy GPT model and generate text     |
| `train`       | pre-train the toy model on `the-verdict.txt`  |
| `load_model`  | load public GPT-2 weights into the toy model  |
| `fine_tuning_classify` | fine-tune GPT-2 for SMS spam classification (both sides) |
| `fine_tuning_instruction` | instruction fine-tuning (Python only; C++ prints a notice) |
| `all`         | run every task in order                       |
| `list`        | list tasks                                    |

Common flags (identical on both sides):

| flag | meaning |
|------|---------|
| `--seed N` | random seed (default: `seed` in `configs/toy_gpt2.json`) |
| `--device D` | force `cpu` / `cuda` / `mps` (default: auto-detect) |
| `--small` | use `model_small` (2 layers, 128 dim) for `toy_model` / `train`; fast on CPU |

Model and training hyper-parameters live in **`configs/toy_gpt2.json`**, read by
both `python_impl/utils/runtime.py` and `cpp_impl/my_torch/utils/runtime.cpp`.
`load_model` / `fine_tuning_*` always use the full GPT-2 shape (they load GPT-2 weights).

## Parity check (Python ↔ C++ numerics)

After building the C++ side:

```shell
python tools/parity_check.py          # small model, ~seconds
python tools/parity_check.py --full   # GPT-2 124M shape
```

Python builds the seeded toy model, hands its weights and a fixed input to the
C++ `build/cpp_impl/tools/parity_check` executable, and compares the logits
(fails if `max |Δ| > 1e-4` or argmax differs).

> Run everything from the repository root: data paths such as `assets/...`
> and the `.temp/` output folders are relative to it.

## Environment Setup

### 1) Python (PyTorch)

```shell
pip install -r requirements.txt   # torch, tiktoken (+ transformers, pandas for load_model / fine_tuning_*)
python main.py            # = python main.py verify
python main.py toy_model
python main.py list
```

### 2) C++ (LibTorch)

Requirements: CMake >= 3.10, a C++20 compiler, git (CMake fetches
`meta-pytorch/tokenizers` on first configure).

Download LibTorch (>= 2.10; older versions lack `torch/headeronly/...`) for your platform from the official PyTorch page
([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)),
e.g. on Linux CPU:

```shell
curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip
unzip libtorch.zip -d cpp_impl/vendor/   # -> cpp_impl/vendor/libtorch
```

Then build and run:

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
# or point at an existing LibTorch: -DLIBTORCH_VENDOR_PATH=/path/to/libtorch
cmake --build build -j

./build/cpp_impl/sandbox/sandbox            # = sandbox verify
./build/cpp_impl/sandbox/sandbox toy_model
./build/cpp_impl/sandbox/sandbox list
```

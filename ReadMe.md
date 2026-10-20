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
- `assets/`: datasets and the GPT-2 tokenizer file shared by both sides
- `main.py`: Python entrypoint
- `cpp_impl/sandbox/main.cpp`: C++ entrypoint (`sandbox` executable)

## Tasks

| task          | what it does                                  |
|---------------|-----------------------------------------------|
| `verify`      | print torch version and selected device (default) |
| `brief`       | torch basic examples                          |
| `toy_model`   | build the toy GPT model and generate text     |
| `train`       | pre-train the toy model on `the-verdict.txt`  |
| `load_model`  | load public GPT-2 weights into the toy model  |
| `fine_tuning` | fine-tuning (Python: instruction, C++: classification) |
| `all`         | run every task in order                       |
| `list`        | list tasks                                    |

> Run everything from the repository root: data paths such as `assets/...`
> and the `.temp/` output folders are relative to it.

## Environment Setup

### 1) Python (PyTorch)

```shell
pip install -r requirements.txt   # torch, tiktoken (+ transformers, pandas for load_model / fine_tuning)
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

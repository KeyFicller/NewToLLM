## LLM from Scratch

This project explores LLM-related implementation from scratch using Torch in both:

- Python (`PyTorch`)
- C++ (`LibTorch`)

## Project Structure

- `python_impl/`: Python-side implementation and verification code
- `cpp_impl/`: C++-side implementation and verification code
- `assets/`: Reference materials and text assets

## Environment Setup

### 1) Python (PyTorch)

Install PyTorch in your active Python environment:

```shell
pip install torch
```

### 2) C++ (LibTorch, macOS)

Download LibTorch from the official PyTorch page:

[https://pytorch.org/get-started/locally/#macos-version](https://pytorch.org/get-started/locally/#macos-version)

Then place the extracted folder under:

`cpp_impl/vendor/libtorch`

## Build and Run

### Python verifier

```shell
python3 main.py
```

### C++ verifier

From the project root:

```shell
cmake -S . -B build
cmake --build build
./build/cpp_impl/sandbox/sandbox
```

## Common Issue

### `Library not loaded: /opt/llvm-openmp/lib/libomp.dylib`

If you hit this error on macOS, run:

```shell
sudo mkdir -p /opt/llvm-openmp/lib
brew install libomp
sudo ln -sf "$(brew --prefix libomp)/lib/libomp.dylib" /opt/llvm-openmp/lib/libomp.dylib
```


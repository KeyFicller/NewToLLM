## LLM from Scratch

This project explores LLM-related implementation from scratch using Torch in both:

- Python (`PyTorch`)
- C++ (`LibTorch`)

## Project Structure

- `python_impl/`: Python-side implementation and verification code
- `cpp_impl/`: C++-side implementation and verification code

## Environment Setup

### 1) Python (PyTorch)

Requirements:

```shell
pip install torch          # Basic
pip install transformers   # Used to Load GPT2
```

### 2) C++ (LibTorch, macOS)

Requirements:

Download LibTorch from the official PyTorch page:

[https://pytorch.org/get-started/locally/#macos-version](https://pytorch.org/get-started/locally/#macos-version)

Then place the extracted folder under:

`cpp_impl/vendor/libtorch`
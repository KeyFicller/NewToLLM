# NewToLLM: a GPT-2 style toy LLM in PyTorch and LibTorch, side by side

The same small GPT (GPT-2 124M shape, plus a tiny `--small` variant) is built from
scratch twice:

- **Python**: `python_impl/` with **PyTorch**
- **C++**: `cpp_impl/my_torch/` with **LibTorch**

Both sides use the same task names, CLI flags, config file and module and parameter names. A parity
check verifies that they compute the same numbers, and a benchmark compares their speed.

## Quick start

```shell
# Python
pip install -r requirements.txt
python main.py list
python main.py toy_model --small

# C++ (LibTorch >= 2.10, CMake >= 3.10, C++20 compiler, git)
curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip
unzip libtorch.zip -d cpp_impl/vendor/          # -> cpp_impl/vendor/libtorch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release  # or -DLIBTORCH_VENDOR_PATH=/path/to/libtorch
cmake --build build -j
./build/cpp_impl/sandbox/sandbox list
./build/cpp_impl/sandbox/sandbox toy_model --small

# Compare
python tools/parity_check.py            # same logits?
python tools/benchmark.py --full        # how fast?
```

> Run everything from the repository root. `assets/...` and the `.temp/` output
> folders are relative to it.

For macOS, download LibTorch from <https://pytorch.org/get-started/locally/>.

## Tasks and status

`python main.py <task>` ↔ `./build/cpp_impl/sandbox/sandbox <task>`

| task | what it does | Python | C++ |
|------|--------------|:------:|:---:|
| `verify` | torch version + selected device (default task) | ✅ | ✅ |
| `brief` | tensor / autograd / tiny MLP training basics | ✅ | ✅ |
| `toy_model` | build the toy GPT, greedy-generate a few tokens | ✅ | ✅ |
| `train` | pre-train on `assets/the-verdict.txt` | ✅ | ✅ |
| `load_model` | load public GPT-2 weights into the toy model | ✅ | ✅ (loads the TorchScript file exported by Python) |
| `fine_tuning_classify` | SMS spam classification fine-tune | ✅ | ✅ |
| `fine_tuning_instruction` | instruction fine-tune | ✅ | ❌ not yet (prints a notice) |
| `all` / `list` | run all tasks / list tasks | ✅ | ✅ |

Flags shared by both entrypoints:

| flag | meaning |
|------|---------|
| `--seed N` | random seed (default: `seed` in `configs/toy_gpt2.json`) |
| `--device D` | force `cpu` / `cuda` / `mps` (default: auto-detect) |
| `--small` | use `model_small` (2 layers, 128 dim) for `toy_model` / `train`; fast on CPU |

Model and training hyper-parameters are stored once, in **`configs/toy_gpt2.json`**. Both
`python_impl/utils/runtime.py` and `cpp_impl/my_torch/utils/runtime.cpp` read it.
`load_model` and `fine_tuning_*` always use the full GPT-2 shape because they load GPT-2 weights.

## File-by-file map

| topic | Python (`python_impl/`) | C++ (`cpp_impl/my_torch/`) |
|-------|-------------------------|----------------------------|
| entrypoint | `../main.py` | `../sandbox/main.cpp` |
| config / runtime | `toy_model/config.py`, `utils/runtime.py` | `toy_model/config.h`, `utils/runtime.{h,cpp}` |
| verify | `verifier/torch_verifier.py` | `verifier/torch_verifier.cpp` |
| basics | `basic/torch_brief.py` | `basic/torch_brief.cpp` |
| model | `toy_model/{transformer,model,torch_toy_model}.py` | `toy_model/{transformer,model,torch_toy_model}.{h,cpp}` |
| pre-train | `train/{data_utils,torch_train}.py` | `train/{data_utils,torch_train}.{h,cpp}` |
| load GPT-2 | `load_model/torch_load_model.py` | `load_model/torch_load_model.cpp` |
| fine-tune | `fine_tuning/{fine_tuning_classify,fine_tuning_instruction,utils}.py` | `fine_tuning/{torch_fine_tuning,fine_tuning_classify}.cpp` |
| device | `utils/torch_utils.py` | `utils/torch_utils.{h,cpp}` |
| parity / benchmark | `../tools/{parity_check,benchmark}.py` | `../tools/{parity_check,benchmark}.cpp` |

Other folders: `configs/` holds the shared config, `assets/` the datasets and GPT-2 tokenizer, and `tests/` the pytest suite.
`extras/python_only/` contains Python-only experiments that are not part of the comparison (see `extras/README.md`).

## Side by side: the same model in both APIs

### Defining a module and registering sub-modules

```python
class ToyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.trf_blocks = nn.Sequential(
            *[ToyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
```

```cpp
struct ToyModelImpl : torch::nn::Module { /* members ... */ };
TORCH_MODULE(ToyModel);   // ToyModel = shared_ptr-like holder of ToyModelImpl

ToyModelImpl::ToyModelImpl(const ToyModelConfig& cfg)
    : tok_emb(register_module("tok_emb",
          torch::nn::Embedding(cfg.vocab_size, cfg.emb_dim))),
      trf_blocks(register_module("trf_blocks", torch::nn::Sequential())) {
  for (int64_t i = 0; i < cfg.n_layers; ++i)
    trf_blocks->push_back(ToyTransformerBlock(cfg));
}
```

### Parameters, buffers and the causal mask

```python
self.scale = nn.Parameter(torch.ones(emb_dim))
self.register_buffer("mask", torch.triu(torch.ones(T, T), diagonal=1))
mask_bool = self.mask[:n_tokens, :n_tokens].to(dtype=torch.bool)
attn_scores.masked_fill_(mask_bool, float("-inf"))
```

```cpp
scale = register_parameter("scale", torch::ones({emb_dim}));
register_buffer("mask", torch::triu(torch::ones({T, T}), 1));
auto mask_bool = named_buffers()["mask"].to(torch::kBool)
                     .index({Slice(None, n_tokens), Slice(None, n_tokens)});
attn_scores.masked_fill_(mask_bool, -std::numeric_limits<float>::infinity());
```

### One training step

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
optimizer.zero_grad()
loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
loss.backward()
optimizer.step()
```

```cpp
torch::optim::AdamW optimizer(model->parameters(),
    torch::optim::AdamWOptions(4e-4).weight_decay(0.1));
optimizer.zero_grad();
auto loss = torch::nn::functional::cross_entropy(
    model->forward(x).flatten(0, 1), y.flatten());
loss.backward();
optimizer.step();
```

### API differences cheat-sheet

| concept | PyTorch | LibTorch |
|---------|---------|----------|
| module definition | subclass `nn.Module` | `struct XImpl : torch::nn::Module` + `TORCH_MODULE(X)` |
| sub-module / parameter / buffer | attribute assignment, `nn.Parameter`, `register_buffer` | must call `register_module` / `register_parameter` / `register_buffer` explicitly |
| access a buffer | `self.mask` | `named_buffers()["mask"]` |
| slicing | `t[:, -k:]` | `t.index({Slice(), Slice(-k, None)})` |
| `nn.Sequential` forward | `seq(x)` | `seq->forward<torch::Tensor>(x)` (typed) |
| shapes | tuples `(B, T)` | brace lists `{B, T}` |
| config | `dict` from JSON | typed `struct ToyModelConfig` filled from the same JSON |
| no-grad | `with torch.no_grad():` | `torch::NoGradGuard guard;` |
| scalar out | `t.item()` | `t.item<float>()` |
| save / load for the other side | `torch.save` / `torch.jit.save` | `torch::pickle_load` / `torch::jit::load` |
| data loading | `Dataset` + `DataLoader` | `torch::data::datasets::Dataset` + `make_data_loader` |

## Parity: do both sides compute the same thing?

```shell
python tools/parity_check.py          # small model
python tools/parity_check.py --full   # GPT-2 124M shape
```

Python builds the seeded model and saves its weights together with a fixed input. The
C++ `parity_check` executable loads them into the C++ `ToyModel` by parameter name and runs
the forward pass. The script then compares the logits and fails if the max |Δ| is above 1e-4
or if the argmax differs. On Linux CPU:

| model | tensors | max abs diff | argmax equal |
|-------|--------:|-------------:|:------------:|
| small | 33 | 8.3e-07 | ✅ |
| full (124M) | 173 | 1.9e-06 | ✅ |

## Benchmark

```shell
python tools/benchmark.py --full --write   # writes docs/benchmark.md
```

Both sides use the same config, seed, batch×tokens and thread count. The script times a no-grad
forward pass and one AdamW step. The latest results are in [docs/benchmark.md](docs/benchmark.md).
On an 8-vCPU Linux box the two sides stay within about 25% of each other (PyTorch 2.14 vs LibTorch 2.10):

| model | side | forward ms | train step ms |
|-------|------|-----------:|--------------:|
| small | python | 18.8 | 81.3 |
| small | cpp | 18.6 | 75.5 |
| full | python | 102.3 | 596.0 |
| full | cpp | 129.2 | 662.0 |

## Tests & CI

```shell
ctest --test-dir build --output-on-failure   # C++ smoke tests
python -m pytest tests                       # Python tests + parity (needs the C++ build)
```

GitHub Actions (`.github/workflows/ci.yml`) builds both sides on CPU with LibTorch 2.10.0.
On every PR it runs CTest, pytest (including the parity check) and the small Python tasks.

## Known gaps

- `fine_tuning_instruction` exists only in Python.
- The `train` data loaders differ slightly. With the same config, Python runs 85 optimizer
  steps and C++ runs 75, so the training curves are not directly comparable yet.
  The model numerics are identical (see parity).
- CI and the numbers above cover Linux CPU only. macOS/MPS and CUDA are untested.

## History

The refactor that produced this layout is documented step by step in
[docs/REFACTOR_PLAN.md](docs/REFACTOR_PLAN.md).

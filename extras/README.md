# extras/

Code that is **not** part of the PyTorch ↔ LibTorch comparison and has no C++
counterpart. It is not wired into `main.py` / `sandbox`.

| file | what it is | how to run (from repo root) |
|------|------------|-----------------------------|
| `python_only/microgpt.py` | dependency-free, pure-Python micro GPT (autograd from scratch), trains on `assets/names.txt` (downloaded on first run) | `python extras/python_only/microgpt.py` |
| `python_only/toy_instruction_eval.py` | tiny local HTTP page (127.0.0.1:8787) to chat with the instruction-fine-tuned toy model; needs `python_impl/.temp/fine-tuned-model-instruction.pth` from `python main.py fine_tuning` | `python extras/python_only/toy_instruction_eval.py` |

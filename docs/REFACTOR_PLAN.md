# NewToLLM 重构计划

目标：把仓库做成一个**清晰、可复现的 PyTorch（Python）与 LibTorch（C++）对照项目**。
同一个实验在两边用同样的名字、同样的配置、同样的数据跑，并且能用脚本检查两边结果是否一致、速度差多少。

原则：每一步都是一个小 PR，合入后仓库仍然可以构建、可以运行；不做大爆炸式重写。

## 现状评估（重构前）

**结构**
- `python_impl/` 与 `cpp_impl/my_torch/` 目录大体一一对应（basic/brief、toy_model、train、load_model、fine_tuning、verifier、utils），这是好的基础。
- 但命名不一致：Python 叫 `basic/torch_brief.py`，C++ 叫 `brief/torch_breif.cpp`（拼写错误）。
- `python_impl/microgpt/microgpt.py`（纯 Python、无依赖的 microgpt）和根目录 `toy_instruction_eval.py`（HTTP 评测页）只有 Python 一侧，也没有接入入口，定位不清。

**两边对等性**
- 模型结构（MultiHeadAttention / LayerNorm / FeedForward / TransformerBlock / ToyModel）和配置（GPT-2 124M）两边基本对齐，参数命名也一致，方便互相加载权重。
- 但同名任务做的事不一样：`fine_tuning` 在 Python 跑的是**指令微调**，在 C++ 跑的是**分类微调**；C++ 没有指令微调。
- 没有统一的随机种子、超参数来源；配置在 `config.py` 和 `config.h` 里各写一份，容易漂移。

**构建 / 运行**
- Python 入口 `main.py` 有 bug：默认/`all` 分支写成了 `args.task == "all" | args.task == "default"`，直接报 `TypeError`；而且启动时就 import 全部模块，缺一个可选依赖整个入口都跑不起来。
- C++ 入口 `sandbox/main.cpp` 没有命令行参数，想换任务只能改代码里的注释再重编。
- LibTorch 路径写死在 `cpp_impl/vendor/libtorch`，README 只写了 macOS。
- 没有 `requirements.txt`；数据和 `.temp/` 路径都是相对仓库根目录的，必须在根目录运行，但文档里没说。

**正确性 / 对比检查**
- 没有任何测试，也没有"两边输出是否一致"的检查。
- 没有性能对比（耗时、吞吐）的统一脚本和结果记录，而这正是这个项目的核心卖点。

## 分步计划

状态：✅ 已完成 / 未标记 = 待做

1. ✅ **统一入口与可运行性**（PR #2，已合入）
   修复 `main.py` 的 bug，并改成任务表 + 延迟 import；C++ `sandbox` 支持命令行参数，任务名与 Python 完全一致（`verify / brief / toy_model / train / load_model / fine_tuning / all / list`）；
   LibTorch 路径可通过 `-DLIBTORCH_VENDOR_PATH` 覆盖；新增 `requirements.txt`；README 补全 Linux/macOS 的构建与运行说明；顺带修复 Linux 下的两个构建问题（`fine_tuning_classify.cpp` 缺 `#include <torch/torch.h>`、静态库 tokenizers 链入共享库需要 `-fPIC`）。
2. ✅ **命名与目录对齐**（PR #3，已合入）
   `brief/torch_breif.cpp` → `basic/torch_brief.cpp`；整理两边目录做到一一对应；把 `microgpt.py`、`toy_instruction_eval.py` 移到 `extras/python_only/`（附 `extras/README.md` 说明），路径改为相对脚本自身解析，不改行为；README 增加两边文件对照表。
3. ✅ **共享配置与随机种子**（PR #4）
   把模型/训练超参数放到一个共享文件（如 `configs/toy_gpt2.json`），两边都从它读取；统一 `--seed`、`--device`、`--small`（小模型，方便 CPU 快速跑）参数。
4. **同名任务做同一件事**
   拆分 `fine_tuning` 为 `fine_tuning_classify` 与 `fine_tuning_instruction`；Python 两个都暴露，C++ 先暴露 classify，指令微调标为 TODO，在 README 的对照表里如实标注。
5. **数值一致性检查（parity）**
   Python 导出一份小模型权重和固定输入 → C++ 加载后前向，比较 logits 最大误差；做成 `tools/parity_check` 脚本 + 小测试，作为"两边实现正确"的硬证据。
6. **性能对比脚本**
   `tools/benchmark`：同配置下两边的前向/训练 step 耗时、tokens/s，输出到 `docs/benchmark.md` 表格（注明硬件与版本）。
7. **测试与 CI**
   Python 用 pytest 跑 `verify`/小模型前向/parity；C++ 用 CTest；GitHub Actions 在 CPU 上构建 LibTorch 版本并运行，保证以后每个 PR 两边都不坏。
8. **README 重写为对照文档**
   每个任务一节：Python 代码 ↔ C++ 代码并排、API 差异说明（如 `register_module`、`Slice`、`forward<>`）、parity 与 benchmark 结果。

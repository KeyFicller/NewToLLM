# NewToLLM 重新设计方案（v2）

> owner 的新要求：代码质量要非常高，对比要非常形象；可以不沿用原有结构。
> 目标：把仓库做成一个示范级的 **PyTorch ↔ LibTorch 并排学习项目**。

## 1. 设计原则

1. **双胞胎文件（twins）**：Python 与 C++ 一一对应，**文件名、类名、函数名完全相同**。
   例如 `py/newtollm/attention.py` 对应 `cpp/newtollm/attention.{h,cpp}`，
   `class MultiHeadAttention` 对应 `class MultiHeadAttentionImpl` 加上 `TORCH_MODULE(MultiHeadAttention)`。
2. **`@twin` 锚点**：每个可以对照的代码块前面都加一行 `# @twin <name>` 或 `// @twin <name>`。
   脚本检查两边的锚点集合与顺序是否一致，不一致就让 CI 失败，这样两边不会悄悄走样。
3. **同一程序、同一输出**：每一课都有 `run_<lesson>()`，两边打印格式完全相同，比如 `x: shape=[2, 3] values=[...]`，保留 4 位小数。
   CI 会把两边的输出逐行比较：文本必须相同，数字误差不超过 1e-4。
4. **没有全局状态**：配置、分词器、随机种子都显式传参。C++ 用 RAII 和 const 引用；Python 用 `@dataclass(frozen=True)` 并写全类型注解。
5. **共享资源**：`configs/presets.json`（tiny / small / gpt2 三档模型和训练超参数）和 `assets/gpt2.tiktoken`（两边都从这个文件离线加载 GPT-2 BPE）。
6. **门禁都是真的**：ruff、mypy --strict、clang-format、clang-tidy、pytest、ctest，以及课程输出 diff、数值 parity、训练曲线 parity、benchmark。
   页面上的数据全部来自工具实际测得的结果文件（JSON），绝不手写。

## 2. 目标结构

```
configs/presets.json            # 共享配置
assets/                         # the-verdict.txt、SMSSpamCollection.tsv、gpt2.tiktoken
py/newtollm/                    # Python 包（PyTorch）
  report.py   config.py  tokenizer.py
  tensors.py  autograd.py  modules.py  attention.py  gpt.py  data.py  train.py  finetune.py
  __main__.py                   # python -m newtollm <lesson>
cpp/newtollm/                   # C++ 库（LibTorch），与上面逐文件对应：*.h + *.cpp
  report  config  tokenizer  tensors  autograd  modules  attention  gpt  data  train  finetune
  main.cpp                      # ./build/newtollm <lesson>
cpp/tools/                      # parity_check.cpp、benchmark.cpp
tools/                          # parity_check.py、benchmark.py、compare_outputs.py、check_twins.py、build_site.py
tests/                          # pytest（同时驱动 C++ 可执行文件）
docs/site/                      # 生成的静态站点（GitHub Pages）
docs/results/                   # 实测结果：parity.json、benchmark.json、train_curves.json
```

## 3. 课程

| # | 课程 | 内容 | 关键 API 差异 |
|---|------|------|---------------|
| 01 | tensors | 创建、形状、view/reshape、广播、matmul、索引 | `x[:, 1:]` ↔ `x.index({Slice(), Slice(1, None)})`，`(2, 3)` ↔ `{2, 3}` |
| 02 | autograd | requires_grad、backward、手写 SGD、no_grad | `with torch.no_grad()` ↔ `torch::NoGradGuard` |
| 03 | modules | GELU / LayerNorm / FeedForward | `nn.Module` ↔ `torch::nn::Module` + `TORCH_MODULE`，`register_parameter/module` |
| 04 | attention | 因果多头注意力，mask 作为 buffer | `register_buffer`，`masked_fill`，`transpose` |
| 05 | gpt | TransformerBlock、GPTModel、贪心生成 | `nn.Sequential` ↔ `->forward<T>()`，`ModuleList` |
| 06 | train | 数据切分、滑动窗口、确定性 batch、AdamW 训练 | `DataLoader` ↔ 手写的确定性 batch 迭代（两边算法相同） |
| 07 | finetune | 垃圾短信分类：换分类头、冻结参数、准确率 | `requires_grad_(False)`，`replace_module` |

**修复 85 步 vs 75 步的差异**：
- 原因一：C++ 端 `Tiktoken()` 默认用的是 llama3 的正则和特殊 token，所以分出来的 token 和 GPT-2 不一样。
- 原因二：两边的 DataLoader 采样方式不同。
- 新设计：两边都用 GPT-2 的正则和 `<|endoftext|>=50256`；窗口用 `unfold` 生成；打乱顺序用 `torch.randperm`（两边使用同一个 ATen CPU 生成器），并且都丢弃最后一个不满的 batch。
- 加一个**训练 parity 门禁**：tiny 配置下，两边每一步的 loss 误差不能超过 1e-4。

## 4. 形象的对比站点（全部由脚本从源码生成）

`tools/build_site.py` 生成 `docs/site/`：
- 每课一个页面：左边 Python、右边 C++，按 `@twin` 块对齐；块内用 difflib 按规范化后的 token 对齐行；
  API 差异（`register_module`、`TORCH_MODULE`、`Slice`、`->`、`{...}` 形状等）高亮显示，悬停可以看到解释；每块配一段"差在哪、为什么"。
- 首页提供 mermaid 图：一张是 GPT 模型结构，一张是 PyTorch ↔ LibTorch 概念映射。
- 结果页：parity 误差表、训练 loss 曲线（两边叠加画）、benchmark 柱状图（SVG），数据全部读自 `docs/results/*.json`。
- `build_site.py --check` 会重新生成一遍，并要求和已提交的版本完全一致，防止页面和源码不同步。

## 5. 步骤（每步一个 PR，合入后仓库可用）

1. **本方案**（只有文档）。
2. **骨架和基础课程**：`pyproject.toml`（ruff/mypy 配置）、`.clang-format`、`.clang-tidy`、根目录 CMake；完成 report/config/tokenizer 以及 01 tensors、02 autograd、03 modules 三课；`check_twins.py` 和 `compare_outputs.py`；pytest、ctest；CI 加入 lint 和课程 diff。旧代码保持不动。
3. **04 attention 和 05 gpt**，加上新的 `parity_check`（前向 logits 对比）。
4. **06 train**：数据管线两边完全一致（token、窗口数、batch 数都相同），训练 parity 门禁，新的 `benchmark`。
5. **07 finetune**（分类微调）。
6. **对比站点生成器**：`docs/site/` 和实测结果 JSON，加 Pages 部署 workflow，再截图。
7. **删除旧的 `python_impl/` 和 `cpp_impl/`**，重写 README，更新本方案的状态。

## 6. 状态

- [x] 1 方案（PR #10）
- [x] 2 骨架和基础课程（PR #11）
- [x] 3 attention 和 gpt
- [x] 4 train（训练步数 288 = 288，逐步 loss 误差 ≤ 2e-6）
- [x] 5 finetune（分类微调，两边逐行一致）
- [x] 6 站点（docs/site，Pages 部署）
- [x] 7 清理旧代码、重写 README

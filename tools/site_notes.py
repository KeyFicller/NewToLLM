"""Hand-written "what differs and why" notes, keyed by ``<module>.<twin block>``.

Blocks without a note get an automatic summary of the API rules they hit.
"""

from __future__ import annotations

LESSON_TITLES: dict[str, tuple[str, str]] = {
    "report": ("00", "统一打印：两边输出逐行可比"),
    "config": ("00", "共享配置：同一份 presets.json"),
    "tokenizer": ("00", "GPT-2 分词器：同一份词表、同一条正则"),
    "tensors": ("01", "张量：创建、形状、索引、广播"),
    "autograd": ("02", "自动求导：梯度、no_grad、手写 SGD"),
    "modules": ("03", "模块：GELU、LayerNorm、FeedForward"),
    "attention": ("04", "因果多头自注意力"),
    "gpt": ("05", "GPT 模型与贪心生成"),
    "data": ("06a", "数据管线：分词、切分、滑窗、batch"),
    "train": ("06b", "预训练：交叉熵、AdamW、loss 曲线"),
    "finetune": ("07", "微调：把 GPT 变成垃圾短信分类器"),
}

NOTES: dict[str, str] = {
    "config.GPTConfig": "Python 用 <code>@dataclass(frozen=True)</code> 得到不可变配置；C++ 用聚合 struct + 指定初始化器（C++20 <code>.name = value</code>）。两边都从同一份 <code>configs/presets.json</code> 读取，没有任何写死在代码里的超参数。",
    "tokenizer.Tokenizer": "Python 用 tiktoken，C++ 用 pytorch/tokenizers 的 <code>Tiktoken</code>；喂给两边的是同一个词表文件、同一条 GPT-2 正则和同一个 <code>&lt;|endoftext|&gt;</code>=50256。C++ 用 pImpl（<code>unique_ptr</code> + 前置声明）把第三方头文件挡在接口之外。",
    "tensors.create": "形状：Python 元组 <code>(2, 3)</code> ↔ C++ 花括号 <code>{2, 3}</code>；dtype：<code>torch.long</code> ↔ <code>torch::kLong</code>。其余函数名几乎一一对应。",
    "tensors.index": "最大的语法差异：C++ 没有切片语法，<code>x[:, 1:]</code> 要写成 <code>x.index({Slice(), Slice(1, None)})</code>。",
    "autograd.no_grad": "<code>with torch.no_grad():</code> 是上下文管理器，C++ 用 RAII：<code>const torch::NoGradGuard guard;</code> 活着的作用域里就不建图。",
    "autograd.fit_line": "手写 SGD。C++ 的 <code>grad()</code> 返回 const 引用，就地清零梯度要用 <code>mutable_grad().zero_()</code>。",
    "modules.LayerNorm": "参数注册：Python 赋值 <code>nn.Parameter</code> 即注册；C++ 必须 <code>register_parameter</code>，否则优化器看不到它。",
    "modules.FeedForward": "Python <code>nn.Sequential(...)</code> ↔ C++ <code>torch::nn::Sequential</code>；C++ 的子模块要 <code>register_module</code>，并用 <code>TORCH_MODULE</code> 生成句柄类。",
    "attention.MultiHeadAttention": "C++ 成员按<strong>声明顺序</strong>初始化（与初始化列表的书写顺序无关），所以成员声明顺序与 Python <code>__init__</code> 完全一致——这正是两边同种子能抽出逐位相同权重的原因。因果 mask 用 <code>register_buffer</code> 注册：随模型走，但不参与训练。",
    "gpt.GPTModel": "<code>nn.Sequential(*[...])</code> ↔ 在初始化列表里调用 <code>make_blocks(cfg)</code>：放在初始化列表而不是构造函数体里，保证随机初始化的顺序与 Python 一致。C++ 的 Sequential 是类型擦除的，所以要写 <code>forward&lt;torch::Tensor&gt;(x)</code>。<code>replace_head</code> 在 C++ 里必须同时更新注册表和成员变量。",
    "gpt.generate_greedy": "负索引切片 <code>idx[:, -context_length:]</code> 在 C++ 里没有直接写法，先算出起点 <code>start</code> 再 <code>Slice(start, None)</code>。",
    "data.split_ids": "按 token 切分而不是按字符：Python 的 <code>len(str)</code> 数码点，C++ 的 <code>std::string::size()</code> 数字节，非 ASCII 文本上两者不同；token 数在两边永远一致。",
    "data.make_windows": "滑动窗口用 <code>unfold(0, context_length, stride)</code> 一行生成，两边 API 完全相同——旧版两边各写一套循环，窗口数不一致，是 85 步 vs 75 步的根源之一。",
    "data.batches": "Python 用生成器 <code>yield</code> 惰性产出；C++ 返回 batch 视图的 <code>std::vector</code>。打乱都用 <code>randperm</code> + 显式生成器，丢弃最后不满的 batch，所以两边 batch 顺序逐位相同。",
    "train.train_model": "训练循环逐行对应：<code>zero_grad → forward → loss → backward → step</code>。AdamW 超参数：Python 关键字参数 ↔ C++ <code>AdamWOptions(lr).weight_decay(wd)</code>。Python 端指定 <code>foreach=False</code>，与 C++ 逐参数更新的实现一致。",
    "train.save_history": "JSON：Python 标准库 <code>json</code> ↔ C++ 的 nlohmann::json（随 tokenizers 一起拉进来），曲线文件格式两边相同，供 parity 门禁和本站使用。",
    "finetune.freeze_body": "按参数名冻结：两边 <code>named_parameters()</code> 的名字完全相同（parity 门禁保证），所以同一组前缀就能选中同一批参数。",
    "finetune.encode_examples": "左侧填充，最后一个位置永远是真实 token，分类直接取最后一位的 logits。Python 切片赋值 ↔ C++ <code>index_put_</code>。",
}

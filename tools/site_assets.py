"""Static pieces of the comparison site: page shell, stylesheet, mermaid diagrams."""

from __future__ import annotations

import html

MERMAID_ARCH = html.escape("""flowchart TD
    ids["token ids [B, T]"] --> tok["tok_emb · Embedding(50257, d)"]
    ids --> pos["pos_emb · Embedding(T, d)"]
    tok --> add(("+"))
    pos --> add
    add --> drop["drop_emb · Dropout"]
    drop --> blocks
    subgraph blocks["trf_blocks · n_layers × TransformerBlock"]
        direction TB
        n1["norm1 · LayerNorm"] --> att["att · MultiHeadAttention<br/>causal mask = register_buffer"]
        att --> r1(("+ shortcut"))
        r1 --> n2["norm2 · LayerNorm"]
        n2 --> ff["ff · FeedForward<br/>Linear → GELU → Linear"]
        ff --> r2(("+ shortcut"))
    end
    blocks --> fn["final_norm · LayerNorm"]
    fn --> head["out_head · Linear(d, 50257)"]
    head --> logits["logits [B, T, 50257]"]
    fn -. "lesson 07: replace_head(2)" .-> cls["Linear(d, 2) · ham / spam"]
""")

MERMAID_CONCEPTS = html.escape("""flowchart LR
    subgraph PY["PyTorch (Python)"]
        p1["class X(nn.Module)"]
        p2["self.fc = nn.Linear(...)"]
        p3["nn.Parameter(t)"]
        p4["with torch.no_grad():"]
        p5["x[:, 1:]"]
        p6["torch.optim.AdamW(params, lr=..)"]
        p7["for x, y in DataLoader"]
        p8["model(x)"]
    end
    subgraph CPP["LibTorch (C++)"]
        c1["class XImpl : torch::nn::Module<br/>TORCH_MODULE(X)"]
        c2["fc_(register_module(&quot;fc&quot;, Linear(..)))"]
        c3["register_parameter(&quot;w&quot;, t)"]
        c4["const torch::NoGradGuard g;"]
        c5["x.index({Slice(), Slice(1, None)})"]
        c6["AdamW(params, AdamWOptions(lr))"]
        c7["for (auto& [x, y] : batches(...))"]
        c8["model->forward(x) / model(x)"]
    end
    p1 <--> c1
    p2 <--> c2
    p3 <--> c3
    p4 <--> c4
    p5 <--> c5
    p6 <--> c6
    p7 <--> c7
    p8 <--> c8
""")

MERMAID_JS = (
    '<script type="module">import mermaid from '
    '"https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";'
    'mermaid.initialize({startOnLoad: true, theme: "dark", securityLevel: "loose"});</script>'
)


def page(title: str, body: str, active: str, nav: list[tuple[str, str, str]]) -> str:
    links = "".join(
        f'<a class="{"on" if key == active else ""}" href="{href}">{html.escape(label)}</a>'
        for key, label, href in nav
    )
    return (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)} · NewToLLM：PyTorch ⇄ LibTorch</title>"
        '<link rel="stylesheet" href="style.css"></head><body>'
        f'<aside class="side"><div class="logo">NewToLLM<small>PyTorch ⇄ LibTorch</small></div>'
        f"{links}</aside><main>{body}"
        '<footer>由 <code>tools/build_site.py</code> 从源码与 <code>docs/results/*.json</code> 生成，'
        "请勿手改。</footer></main>"
        f"{MERMAID_JS}</body></html>\n"
    )


CSS = """
:root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e;
  --py:#3572A5; --py2:#79c0ff; --cpp:#f34b7d; --cpp2:#ffa198; --api:#e3b341; --same:#1c2a1f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 -apple-system,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif; display:flex; }
a { color:#58a6ff; text-decoration:none; } a:hover { text-decoration:underline; }
code { font-family:"JetBrains Mono","Fira Code",Menlo,Consolas,monospace; font-size:.9em; background:#262c36; padding:.1em .35em; border-radius:4px; }
.side { position:sticky; top:0; height:100vh; width:210px; flex:none; background:var(--panel); border-right:1px solid var(--line); padding:16px 10px; overflow:auto; }
.side a { display:block; padding:5px 10px; border-radius:6px; color:var(--text); font-size:14px; }
.side a.on { background:#1f6feb33; color:#fff; font-weight:600; }
.logo { font-weight:800; font-size:20px; padding:0 10px 14px; }
.logo small { display:block; font-weight:400; font-size:12px; color:var(--muted); }
main { flex:1; min-width:0; padding:24px 32px 60px; }
h1 { margin:.2em 0; } h2 { margin-top:2em; border-bottom:1px solid var(--line); padding-bottom:.3em; }
.hero h1 { font-size:40px; background:linear-gradient(90deg,var(--py2),var(--cpp)); -webkit-background-clip:text; color:transparent; }
.hero p { max-width:900px; color:#c9d1d9; font-size:16px; }
.cards { display:flex; gap:14px; flex-wrap:wrap; margin:20px 0; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 18px; min-width:170px; color:var(--muted); }
.card .big { font-size:28px; font-weight:800; color:var(--text); }
table.lessons { border-collapse:collapse; width:100%; }
table.lessons td, table.lessons th { border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; }
td.r { text-align:right; font-variant-numeric:tabular-nums; }
.mermaid-wrap { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px; overflow:auto; }
.lesson-head { display:flex; gap:18px; align-items:center; }
.lesson-head .num { font-size:44px; font-weight:900; color:var(--api); }
.files, .stats, .meta { color:var(--muted); margin:2px 0; }
.toc { position:sticky; top:0; z-index:5; background:var(--bg); padding:8px 0; border-bottom:1px solid var(--line); display:flex; flex-wrap:wrap; gap:6px; }
.toc a { font-size:13px; background:var(--panel); border:1px solid var(--line); border-radius:20px; padding:2px 10px; color:var(--text); }
.block { margin:26px 0; }
.block h3 { margin:0 0 6px; font-family:monospace; font-size:17px; }
.twin { color:var(--api); font-weight:400; }
.anchor { margin-left:8px; color:var(--muted); }
.note { background:#1f2a3a; border-left:3px solid #58a6ff; padding:8px 12px; border-radius:4px; margin:6px 0 10px; color:#c9d1d9; }
.badge { display:inline-block; font-size:12px; font-weight:700; padding:1px 8px; border-radius:10px; color:#fff; margin-right:6px; }
.badge.py { background:var(--py); } .badge.cpp { background:var(--cpp); }
table.twin-table { border-collapse:collapse; width:100%; table-layout:fixed; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
table.twin-table th { text-align:left; font-weight:500; color:var(--muted); font-size:13px; padding:6px 10px; border-bottom:1px solid var(--line); background:#1b2129; }
col.lnc { width:42px; } col.codec { width:calc(50% - 42px); }
td.ln { color:#6e7681; text-align:right; padding:0 8px; font:12px/1.55 monospace; user-select:none; vertical-align:top; border-right:1px solid #21262d; }
td.code { font:13px/1.55 "JetBrains Mono","Fira Code",Menlo,Consolas,monospace; white-space:pre-wrap; word-break:break-word; padding:0 10px; vertical-align:top; }
td.code.cpp { border-left:2px solid #f34b7d55; } td.code.py { border-left:2px solid #3572a555; }
td.gap { background:repeating-linear-gradient(135deg,#161b22 0 6px,#1c2128 6px 12px); }
tr.same td.code { background:var(--same); }
td.has-api { background:#2d2611; }
mark.api { background:#e3b34133; color:inherit; border-bottom:2px solid var(--api); border-radius:3px; cursor:help; }
.legend.same { background:var(--same); padding:0 6px; border-radius:3px; }
.transcript td.code { font-size:12px; } .transcript tr.num td.code { background:#2b1d24; }
.verdict { font-size:12px; margin-left:10px; padding:2px 8px; border-radius:10px; font-family:sans-serif; }
.verdict.ok { background:#238636; } .verdict.bad { background:#da3633; }
.chart { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:10px; margin:12px 0; }
.chart text { fill:var(--muted); font-size:12px; font-family:sans-serif; }
.chart .title { fill:var(--text); font-size:14px; font-weight:600; }
footer { margin-top:50px; color:var(--muted); font-size:12px; }
.howto li { margin:6px 0; }
/* pygments (short class names) */
.k,.kd,.kn,.kr,.kt,.kc,.kp { color:#ff7b72; } .nf,.fm { color:#d2a8ff; } .nc { color:#f0883e; }
.s,.s1,.s2,.sa,.sd,.se,.si,.sb,.dl,.sc { color:#a5d6ff; } .sd { color:#8b949e; }
.c,.c1,.cm,.cs,.ch,.cpf { color:#8b949e; font-style:italic; } .cp { color:#ff7b72; }
.m,.mi,.mf,.mh,.il { color:#79c0ff; } .o,.ow { color:#ff7b72; } .nb,.bp { color:#ffa657; }
.nd { color:#d2a8ff; } .nn { color:#f0883e; }
"""

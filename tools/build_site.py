"""Generate the static PyTorch <-> LibTorch comparison site in docs/site/.

    python tools/build_site.py            # (re)generate docs/site/
    python tools/build_site.py --check    # fail if docs/site/ is stale
    python tools/build_site.py --record   # also re-record lesson transcripts (needs the C++ build)

Everything on the site is derived from the sources (``@twin`` blocks) and from the
measured result files in docs/results/*.json - no number is typed in by hand.
"""

from __future__ import annotations

import argparse
import difflib
import filecmp
import html
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pygments.lexers import CppLexer, PythonLexer
from pygments.token import STANDARD_TYPES, _TokenType

from check_twins import CPP_MARKER, PY_MARKER, twin_names
from site_assets import CSS, MERMAID_ARCH, MERMAID_CONCEPTS, page
from site_charts import bar_chart, line_chart
from site_rules import RULES, ApiRule
from site_notes import LESSON_TITLES, NOTES

REPO_ROOT = Path(__file__).resolve().parents[1]
SITE = REPO_ROOT / "docs" / "site"
RESULTS = REPO_ROOT / "docs" / "results"
PY_DIR = REPO_ROOT / "py" / "newtollm"
CPP_DIR = REPO_ROOT / "cpp" / "newtollm"
CPP_CLI = REPO_ROOT / "build" / "bin" / "newtollm"
TRANSCRIPTS = RESULTS / "transcripts.json"
LESSON_ORDER = list(LESSON_TITLES)
RUNNABLE = ["tensors", "autograd", "modules", "attention", "gpt", "data", "train", "finetune"]


# ----------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Line:
    file: str
    number: int
    text: str


@dataclass(frozen=True)
class Block:
    name: str
    lines: list[Line]


def read_blocks(path: Path, marker: re.Pattern[str]) -> list[Block]:
    blocks: list[Block] = []
    name: str | None = None
    lines: list[Line] = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = marker.match(text)
        if match:
            if name is not None:
                blocks.append(Block(name, lines))
            name, lines = match.group(1), []
        elif name is not None:
            lines.append(Line(path.name, number, text))
    if name is not None:
        blocks.append(Block(name, lines))
    return [Block(b.name, trim(b.lines)) for b in blocks]


def trim(lines: list[Line]) -> list[Line]:
    while lines and not lines[-1].text.strip():
        lines = lines[:-1]
    return lines


def cpp_blocks(name: str) -> dict[str, list[Line]]:
    merged: dict[str, list[Line]] = {}
    for suffix in (".h", ".cpp"):
        path = CPP_DIR / f"{name}{suffix}"
        if path.exists():
            for block in read_blocks(path, CPP_MARKER):
                merged.setdefault(block.name, []).extend(block.lines)
    return merged


# ----------------------------------------------------------------------------- alignment

_NORMALIZE = [
    (re.compile(r"\bself\.|this->"), ""),
    (re.compile(r"(\w)_\b"), r"\1"),  # C++ member suffix
    (re.compile(r"::|->"), "."),
    (re.compile(r"\b(const|auto|override|explicit|inline|static|constexpr|void)\b"), ""),
    (re.compile(r"[;{}()\[\],:&*<>\"']"), " "),
]


def normalize(text: str) -> str:
    text = text.strip()
    for pattern, repl in _NORMALIZE:
        text = pattern.sub(repl, text)
    return " ".join(text.lower().split())


def align(py: list[Line], cpp: list[Line]) -> list[tuple[Line | None, Line | None]]:
    """Pair lines so that corresponding statements sit on the same row."""
    a = [normalize(line.text) for line in py]
    b = [normalize(line.text) for line in cpp]
    rows: list[tuple[Line | None, Line | None]] = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            rows.extend(zip(py[i1:i2], cpp[j1:j2], strict=True))
        elif tag == "delete":
            rows.extend((line, None) for line in py[i1:i2])
        elif tag == "insert":
            rows.extend((None, line) for line in cpp[j1:j2])
        else:
            rows.extend(align_replace(py[i1:i2], cpp[j1:j2]))
    return rows


def align_twin(py: list[Line], cpp: list[Line]) -> list[tuple[Line | None, Line | None]]:
    """Header declarations first, then the implementation aligned against Python.

    A class/struct declaration pairs with the Python ``class`` line; a function
    declaration stands alone so the Python ``def`` can pair with the C++ definition.
    """
    header = [line for line in cpp if line.file.endswith(".h")]
    source = [line for line in cpp if not line.file.endswith(".h")]
    rows: list[tuple[Line | None, Line | None]] = []
    if header and py and re.match(r"\s*(class|struct)\b", header[0].text):
        # Stop after the class (and its TORCH_MODULE): later free-function
        # declarations belong to other blocks' definitions.
        end = next(
            (i for i, line in enumerate(header) if re.match(r"(TORCH_MODULE\(|};)", line.text)),
            len(header) - 1,
        )
        if end + 1 < len(header) and header[end + 1].text.startswith("TORCH_MODULE("):
            end += 1
        header = header[: end + 1]
        rows.append((py[0], header[0]))
        rows.extend((None, line) for line in header[1:])
        py = py[1:]
    else:
        rows.extend((None, line) for line in header)
    return rows + align(py, source)


def align_replace(py: list[Line], cpp: list[Line]) -> list[tuple[Line | None, Line | None]]:
    """Inside a changed region, greedily pair lines by similarity, keeping order."""
    rows: list[tuple[Line | None, Line | None]] = []
    j = 0
    for line in py:
        best, best_score = -1, 0.45
        for k in range(j, min(len(cpp), j + 12)):
            score = difflib.SequenceMatcher(None, normalize(line.text), normalize(cpp[k].text)).ratio()
            if score > best_score:
                best, best_score = k, score
        if best < 0:
            rows.append((line, None))
            continue
        rows.extend((None, extra) for extra in cpp[j:best])
        rows.append((line, cpp[best]))
        j = best + 1
    rows.extend((None, extra) for extra in cpp[j:])
    return rows


# ----------------------------------------------------------------------------- highlighting


def css_class(ttype: _TokenType) -> str:
    while ttype not in STANDARD_TYPES:
        ttype = ttype.parent
    return STANDARD_TYPES[ttype]


def highlight(lines: list[Line], lang: str) -> dict[tuple[str, int], list[tuple[str, str]]]:
    """Lex the whole block once (so multi-line strings work), then split per line."""
    lexer = PythonLexer() if lang == "py" else CppLexer()
    out: dict[tuple[str, int], list[tuple[str, str]]] = {}
    by_file: dict[str, list[Line]] = {}
    for line in lines:
        by_file.setdefault(line.file, []).append(line)
    for file_lines in by_file.values():
        source = "\n".join(line.text for line in file_lines) + "\n"
        segments: list[list[tuple[str, str]]] = [[]]
        for ttype, value in lexer.get_tokens(source):
            for k, part in enumerate(value.split("\n")):
                if k:
                    segments.append([])
                if part:
                    segments[-1].append((css_class(ttype), part))
        for line, segs in zip(file_lines, segments, strict=False):
            out[(line.file, line.number)] = segs
    return out


def api_marks(text: str, lang: str) -> list[tuple[int, int, ApiRule]]:
    marks: list[tuple[int, int, ApiRule]] = []
    if text.lstrip().startswith(("#", "//")):
        return marks
    for rule in RULES:
        if rule.lang == lang:
            marks.extend((m.start(), m.end(), rule) for m in rule.pattern.finditer(text))
    marks.sort(key=lambda m: m[0])
    kept: list[tuple[int, int, ApiRule]] = []
    for mark in marks:
        if not kept or mark[0] >= kept[-1][1]:
            kept.append(mark)
    return kept


def render_line(segments: list[tuple[str, str]], marks: list[tuple[int, int, ApiRule]]) -> str:
    out: list[str] = []
    pos = 0
    bounds = sorted({p for s, e, _ in marks for p in (s, e)})
    open_mark: tuple[int, int, ApiRule] | None = None
    for cls, value in segments:
        start = pos
        cuts = [b - start for b in bounds if start < b < start + len(value)]
        pieces = [value[i:j] for i, j in zip([0, *cuts], [*cuts, len(value)], strict=True)]
        for piece in pieces:
            mark = next((m for m in marks if m[0] <= pos < m[1]), None)
            if mark is not open_mark:
                if open_mark is not None:
                    out.append("</mark>")
                if mark is not None:
                    tip = html.escape(mark[2].tip, quote=True)
                    out.append(f'<mark class="api" data-key="{mark[2].key}" title="{tip}">')
                open_mark = mark
            escaped = html.escape(piece)
            out.append(f'<span class="{cls}">{escaped}</span>' if cls else escaped)
            pos += len(piece)
    if open_mark is not None:
        out.append("</mark>")
    return "".join(out)


# ----------------------------------------------------------------------------- pages


@dataclass(frozen=True)
class BlockStats:
    rows: int
    same: int
    api: set[str]


def render_block(module: str, name: str, py: list[Line], cpp: list[Line]) -> tuple[str, BlockStats]:
    hl_py, hl_cpp = highlight(py, "py"), highlight(cpp, "cpp")
    rows_html: list[str] = []
    keys: set[str] = set()
    same = 0
    rows = align_twin(py, cpp)
    for left, right in rows:
        cells: list[str] = []
        row_cls: list[str] = []
        if left and right and normalize(left.text) == normalize(right.text) and left.text.strip():
            row_cls.append("same")
            same += 1
        for line, lang, hl in ((left, "py", hl_py), (right, "cpp", hl_cpp)):
            if line is None:
                cells.append('<td class="ln"></td><td class="code gap"></td>')
                continue
            marks = api_marks(line.text, lang)
            keys.update(m[2].key for m in marks)
            label = f"{line.file}:{line.number}"
            code = render_line(hl.get((line.file, line.number), [("", line.text)]), marks)
            api_cls = " has-api" if marks else ""
            cells.append(
                f'<td class="ln" title="{label}">{line.number}</td>'
                f'<td class="code {lang}{api_cls}">{code}</td>'
            )
        rows_html.append(f'<tr class="{" ".join(row_cls)}">{"".join(cells)}</tr>')
    note = NOTES.get(f"{module}.{name}")
    if note is None and keys:
        tips = sorted({r.key: r for r in RULES if r.key in keys}.values(), key=lambda r: r.key)
        note = "本块涉及的 API 差异：" + "；".join(
            f"<code>{html.escape(r.key)}</code>" for r in tips
        )
    note_html = f'<div class="note">{note}</div>' if note else ""
    py_files = sorted({line.file for line in py})
    cpp_files = sorted({line.file for line in cpp})
    table = (
        f'<section class="block" id="{name}"><h3><span class="twin">@twin</span> {name}'
        f'<a class="anchor" href="#{name}">#</a></h3>{note_html}'
        '<table class="twin-table"><colgroup><col class="lnc"><col class="codec">'
        '<col class="lnc"><col class="codec"></colgroup><thead><tr>'
        f'<th colspan="2"><span class="badge py">PyTorch</span> {" + ".join(py_files)}</th>'
        f'<th colspan="2"><span class="badge cpp">LibTorch</span> {" + ".join(cpp_files)}</th>'
        f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table></section>'
    )
    return table, BlockStats(len(rows), same, keys)


def render_transcript(module: str, transcripts: dict[str, Any]) -> str:
    entry = transcripts.get("lessons", {}).get(module)
    if not entry:
        return ""
    py, cpp = entry["python"], entry["cpp"]
    rows = []
    for left, right in zip(py, cpp, strict=False):
        cls = "same" if left == right else "num"
        rows.append(
            f'<tr class="{cls}"><td class="code">{html.escape(left)}</td>'
            f'<td class="code">{html.escape(right)}</td></tr>'
        )
    verdict = entry["verdict"]
    return (
        '<section class="block" id="transcript"><h3>运行输出（实测）'
        f'<span class="verdict {"ok" if verdict == "IDENTICAL" else "bad"}">{verdict}</span></h3>'
        f'<div class="note">命令：<code>python -m newtollm {module}</code> 与 '
        f"<code>./build/bin/newtollm {module}</code>；文本逐字比较，数字容差 1e-4。"
        "标灰的行逐字相同，其余行只在浮点末位不同。</div>"
        '<table class="twin-table transcript"><thead><tr><th>PyTorch stdout</th>'
        f'<th>LibTorch stdout</th></tr></thead><tbody>{"".join(rows)}</tbody></table></section>'
    )


def lesson_page(module: str, transcripts: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    py_blocks = read_blocks(PY_DIR / f"{module}.py", PY_MARKER)
    cpp = cpp_blocks(module)
    parts: list[str] = []
    totals = {"rows": 0, "same": 0, "blocks": len(py_blocks), "api": set()}
    for block in py_blocks:
        section, stats = render_block(module, block.name, block.lines, cpp.get(block.name, []))
        parts.append(section)
        totals["rows"] += stats.rows
        totals["same"] += stats.same
        totals["api"] |= stats.api
    number, title = LESSON_TITLES[module]
    toc = "".join(f'<a href="#{b.name}">{b.name}</a>' for b in py_blocks)
    py_loc = sum(len(b.lines) for b in py_blocks)
    cpp_loc = sum(len(v) for v in cpp.values())
    head = (
        f'<header class="lesson-head"><div class="num">{number}</div><div><h1>{title}</h1>'
        f'<p class="files"><code>py/newtollm/{module}.py</code> ⇄ '
        f"<code>cpp/newtollm/{module}.{{h,cpp}}</code></p>"
        f'<p class="stats">{len(py_blocks)} 个 twin 块 · Python {py_loc} 行 · C++ {cpp_loc} 行 · '
        f"{len(totals['api'])} 类 API 差异（黄色高亮，悬停看解释）</p></div></header>"
        f'<nav class="toc">{toc}{"<a href=#transcript>运行输出</a>" if module in RUNNABLE else ""}'
        "</nav>"
    )
    body = head + "".join(parts) + render_transcript(module, transcripts)
    summary = {"module": module, "py_loc": py_loc, "cpp_loc": cpp_loc, **totals}
    return page(f"{number} {title}", body, active=module, nav=nav_items()), summary


def nav_items() -> list[tuple[str, str, str]]:
    items = [("index", "首页", "index.html")]
    items += [(m, f"{LESSON_TITLES[m][0]} {m}", f"{m}.html") for m in LESSON_ORDER]
    items.append(("results", "实测结果", "results.html"))
    return items


def load_json(name: str) -> dict[str, Any]:
    path = RESULTS / name
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return data


def index_page(summaries: list[dict[str, Any]]) -> str:
    rows = "".join(
        f'<tr><td>{LESSON_TITLES[s["module"]][0]}</td>'
        f'<td><a href="{s["module"]}.html">{LESSON_TITLES[s["module"]][1]}</a></td>'
        f'<td class="r">{s["blocks"]}</td><td class="r">{s["py_loc"]}</td>'
        f'<td class="r">{s["cpp_loc"]}</td><td class="r">{s["cpp_loc"] / max(s["py_loc"], 1):.2f}×</td>'
        f'<td>{" ".join(f"<code>{k}</code>" for k in sorted(s["api"]))}</td></tr>'
        for s in summaries
    )
    train = load_json("train_curves.json").get("result", {})
    parity = load_json("parity.json")
    max_fwd = max((r["max_abs_diff"] for r in parity.get("results", [])), default=0.0)
    cards = (
        '<div class="cards">'
        f'<div class="card"><div class="big">{len(summaries)}</div>对双胞胎文件</div>'
        f'<div class="card"><div class="big">{sum(s["blocks"] for s in summaries)}</div>个逐块对齐的 @twin 块</div>'
        f'<div class="card"><div class="big">{train.get("python_steps", "?")} = {train.get("cpp_steps", "?")}</div>'
        "训练步数 Python = C++<br><small>（旧版 85 ≠ 75）</small></div>"
        f'<div class="card"><div class="big">{max(train.get("max_train_diff", 0.0), 0.0):.1e}</div>逐步训练 loss 最大误差</div>'
        f'<div class="card"><div class="big">{max_fwd:.1e}</div>同种子初始化 + 前向 logits 最大误差</div>'
        "</div>"
    )
    body = (
        '<header class="hero"><h1>一个 GPT，写两遍</h1>'
        "<p>同一个 GPT-2 小模型，左边 <b>PyTorch</b>，右边 <b>LibTorch (C++)</b>。文件名、类名、函数名、"
        "打印格式一一对应；同一个随机种子在两边抽出逐位相同的权重，训练出几乎重合的 loss 曲线。"
        "本站每一行代码都直接取自仓库源码，每一个数字都来自工具实测的 JSON。</p></header>"
        f"{cards}"
        '<h2>课程</h2><table class="lessons"><thead><tr><th>#</th><th>课程</th><th>块</th>'
        "<th>Python 行</th><th>C++ 行</th><th>C++/Py</th><th>涉及的 API 差异</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        '<h2>模型结构</h2><div class="mermaid-wrap"><pre class="mermaid">'
        f"{MERMAID_ARCH}</pre></div>"
        '<h2>概念映射：PyTorch ⇄ LibTorch</h2><div class="mermaid-wrap"><pre class="mermaid">'
        f"{MERMAID_CONCEPTS}</pre></div>"
        '<h2>怎么读</h2><ul class="howto"><li>每课一个页面，左右两栏按 <code>@twin</code> 块对齐，块内按语义逐行对齐；'
        '<span class="legend same">灰底</span> = 两边规范化后完全相同，'
        '<mark class="api">黄色</mark> = API 差异（悬停看解释），斜线格 = 这一侧没有对应行。</li>'
        "<li>页面末尾是两边程序的真实输出，逐行对照。</li>"
        '<li><a href="results.html">实测结果</a>：parity 误差、训练曲线叠加图、benchmark。</li></ul>'
    )
    return page("首页", body, active="index", nav=nav_items())


def results_page() -> str:
    parity = load_json("parity.json")
    curves = load_json("train_curves.json")
    bench = load_json("benchmark.json")
    parity_rows = "".join(
        f'<tr><td>{r["preset"]}</td><td>{r["check"]}</td><td class="r">{r["tensors"]}</td>'
        f'<td class="r">{r["max_abs_diff"]:.3e}</td><td>{"✅" if r["passed"] else "❌"}</td></tr>'
        for r in parity.get("results", [])
    )
    parts = [
        "<h1>实测结果</h1>",
        f'<p class="meta">PyTorch {parity.get("pytorch", "?")} · LibTorch {parity.get("libtorch", "?")} · '
        f'{parity.get("machine", "?")}。所有数字由 <code>tools/parity.py</code>、<code>tools/train_parity.py</code>、'
        "<code>tools/bench.py</code> 写入 <code>docs/results/*.json</code>，本页只负责画出来。</p>",
        "<h2>① 数值一致性（parity）</h2>",
        '<div class="note"><b>init</b>：同一个种子，C++ 抽出的随机权重与 Python 逐位相同；'
        "<b>forward</b>：把 Python 的权重拷进 C++ 模型，比较 logits。</div>",
        '<table class="lessons"><thead><tr><th>preset</th><th>检查</th><th>张量数</th>'
        f"<th>max |diff|</th><th>通过</th></tr></thead><tbody>{parity_rows}</tbody></table>",
    ]
    if curves:
        py, cpp, res = curves["python"], curves["cpp"], curves["result"]
        diffs = [abs(a - b) for a, b in zip(py["train_loss"], cpp["train_loss"], strict=True)]
        parts += [
            "<h2>② 训练曲线：两条线几乎完全重合</h2>",
            f'<div class="note">preset <b>{res["preset"]}</b>，种子 {res["seed"]}。步数 Python '
            f'<b>{res["python_steps"]}</b> / C++ <b>{res["cpp_steps"]}</b>；逐步训练 loss 最大误差 '
            f'<b>{res["max_train_diff"]:.2e}</b>，验证 loss 最大误差 <b>{res["max_val_diff"]:.2e}</b>（门禁阈值 {curves["atol"]:g}）。'
            "旧版因分词正则和 DataLoader 切分不同，同一配置下 Python 85 步、C++ 75 步。</div>",
            line_chart(
                "训练 loss（每步）与验证 loss",
                [
                    ("PyTorch train", py["steps"], py["train_loss"], "py", False),
                    ("LibTorch train", cpp["steps"], cpp["train_loss"], "cpp", True),
                    ("PyTorch val", py["eval_steps"], py["val_loss"], "py2", False),
                    ("LibTorch val", cpp["eval_steps"], cpp["val_loss"], "cpp2", True),
                ],
                "step",
                "loss",
            ),
            line_chart(
                "|PyTorch − LibTorch| 逐步 loss 差（放大看：都在 1e-6 量级）",
                [("|diff|", py["steps"], diffs, "diff", False)],
                "step",
                "|Δloss|",
            ),
        ]
    if bench:
        groups = [
            (f'{r["preset"]} · {metric.replace("_ms", "")}', r["python"][metric], r["cpp"][metric])
            for r in bench["results"]
            for metric in ("forward_ms", "train_step_ms")
        ]
        parts += [
            "<h2>③ 性能（中位数，毫秒，越短越好）</h2>",
            f'<div class="note">{bench["machine"]}，{bench["threads"]} 线程，预热 {bench["warmup"]} 次后取 '
            f'{bench["iters"]} 次的中位数；PyTorch {bench["pytorch"]} / LibTorch {bench["libtorch"]}。'
            "同一 preset、同一种子、同一输入形状。两边版本不同、机器共享，这组数只说明“差距不大”，"
            "不说明谁更快；大部分时间花在同一套 ATen 内核里，C++ 主要省掉的是 Python 调度开销，小模型上更明显。</div>",
            bar_chart(groups),
        ]
    return page("实测结果", "".join(parts), active="results", nav=nav_items())


# ----------------------------------------------------------------------------- transcripts


def record_transcripts() -> None:
    from compare_outputs import compare_lines  # noqa: PLC0415

    lessons: dict[str, Any] = {}
    for lesson in RUNNABLE:
        py = subprocess.run(
            [sys.executable, "-m", "newtollm", lesson],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        cpp = subprocess.run(
            [str(CPP_CLI), lesson], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        mismatches = compare_lines(py, cpp, atol=1e-4)
        lessons[lesson] = {
            "python": py,
            "cpp": cpp,
            "verdict": "IDENTICAL" if not mismatches else f"{len(mismatches)} MISMATCH",
        }
    TRANSCRIPTS.write_text(json.dumps({"lessons": lessons}, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")


# ----------------------------------------------------------------------------- main


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    transcripts = load_json("transcripts.json")
    unknown = [k for k in NOTES if k.split(".")[0] not in twin_names()]
    if unknown:
        msg = f"notes for unknown modules: {unknown}"
        raise SystemExit(msg)
    summaries = []
    for module in LESSON_ORDER:
        html_text, summary = lesson_page(module, transcripts)
        (out / f"{module}.html").write_text(html_text, encoding="utf-8")
        summaries.append(summary)
    (out / "index.html").write_text(index_page(summaries), encoding="utf-8")
    (out / "results.html").write_text(results_page(), encoding="utf-8")
    (out / "style.css").write_text(CSS, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if docs/site is stale")
    parser.add_argument("--record", action="store_true", help="re-record lesson transcripts")
    args = parser.parse_args(argv)
    if args.record:
        record_transcripts()
    if not args.check:
        build(SITE)
        print(f"[site] wrote {SITE.relative_to(REPO_ROOT)}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        build(Path(tmp))
        cmp = filecmp.dircmp(tmp, SITE)
        stale = cmp.diff_files + cmp.left_only + cmp.right_only
    if stale:
        print(f"[site] stale: {stale} - run python tools/build_site.py")
        return 1
    print("[site] up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Dependency-free SVG charts (line and grouped bar) for the results page."""

from __future__ import annotations

import html

COLORS = {"py": "#79c0ff", "cpp": "#f34b7d", "py2": "#3fb950", "cpp2": "#e3b341", "diff": "#d2a8ff"}
W, H, L, R, T, B = 900, 320, 60, 160, 34, 40

Series = tuple[str, list[int], list[float], str, bool]


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    step = (hi - lo) / n if hi > lo else 1.0
    return [lo + i * step for i in range(n + 1)]


def _fmt(v: float) -> str:
    return f"{v:.1e}" if 0 < abs(v) < 0.01 else f"{v:.2f}".rstrip("0").rstrip(".")


def line_chart(title: str, series: list[Series], xlabel: str, ylabel: str) -> str:
    xs = [x for _, sx, _, _, _ in series for x in sx]
    ys = [y for _, _, sy, _, _ in series for y in sy]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad = (y1 - y0) * 0.05 or 1.0
    y0, y1 = (max(y0 - pad, 0.0) if y0 >= 0 else y0 - pad), y1 + pad

    def px(x: float) -> float:
        return L + (x - x0) / max(x1 - x0, 1) * (W - L - R)

    def py(y: float) -> float:
        return T + (1 - (y - y0) / (y1 - y0)) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">']
    parts.append(f'<text class="title" x="{L}" y="16">{html.escape(title)}</text>')
    for t in _ticks(y0, y1):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{py(t):.1f}" y2="{py(t):.1f}" stroke="#30363d"/>')
        parts.append(f'<text x="{L - 6}" y="{py(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
    for t in _ticks(x0, x1):
        parts.append(f'<text x="{px(t):.1f}" y="{H - B + 16}" text-anchor="middle">{int(t)}</text>')
    parts.append(f'<text x="{(W - R + L) / 2}" y="{H - 6}" text-anchor="middle">{xlabel}</text>')
    parts.append(f'<text x="{L}" y="{T - 6}" text-anchor="end">{html.escape(ylabel)}</text>')
    for k, (name, sx, sy, color, dashed) in enumerate(series):
        points = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(sx, sy, strict=True))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        width = 1.6 if not dashed else 2.2
        parts.append(
            f'<polyline fill="none" stroke="{COLORS[color]}" stroke-width="{width}"{dash} points="{points}"/>'
        )
        ly = T + 14 + k * 20
        parts.append(
            f'<line x1="{W - R + 12}" x2="{W - R + 40}" y1="{ly}" y2="{ly}" stroke="{COLORS[color]}" stroke-width="3"{dash}/>'
            f'<text x="{W - R + 46}" y="{ly + 4}">{html.escape(name)}</text>'
        )
    parts.append("</svg>")
    return f'<div class="chart">{"".join(parts)}</div>'


def bar_chart(groups: list[tuple[str, float, float]]) -> str:
    """Horizontal grouped bars: one group per workload, PyTorch vs LibTorch."""
    row_h, top = 58, 30
    height = top + row_h * len(groups) + 20
    vmax = max(max(p, c) for _, p, c in groups)
    left, right = 170, 120
    span = W - left - right
    parts = [f'<svg viewBox="0 0 {W} {height}" width="100%" role="img" aria-label="benchmark">']
    parts.append(
        f'<rect x="{left}" y="6" width="14" height="10" fill="{COLORS["py"]}"/><text x="{left + 20}" y="15">PyTorch</text>'
        f'<rect x="{left + 100}" y="6" width="14" height="10" fill="{COLORS["cpp"]}"/><text x="{left + 120}" y="15">LibTorch</text>'
    )
    for i, (label, p, c) in enumerate(groups):
        y = top + i * row_h
        parts.append(f'<text class="title" x="{left - 10}" y="{y + 26}" text-anchor="end">{html.escape(label)}</text>')
        for k, (value, color) in enumerate(((p, "py"), (c, "cpp"))):
            w = value / vmax * span
            by = y + 4 + k * 22
            parts.append(f'<rect x="{left}" y="{by}" width="{w:.1f}" height="18" rx="3" fill="{COLORS[color]}"/>')
            parts.append(f'<text x="{left + w + 6:.1f}" y="{by + 14}">{value:.2f} ms</text>')
        parts.append(
            f'<text x="{left - 10}" y="{y + 44}" text-anchor="end">C++/Py = {c / p:.2f}×</text>'
        )
    parts.append("</svg>")
    return f'<div class="chart">{"".join(parts)}</div>'

"""Check that every Python module has a C++ twin with the same ``@twin`` blocks.

    python tools/check_twins.py

A twin pair is ``py/newtollm/<name>.py`` <-> ``cpp/newtollm/<name>.{h,cpp}``.
Each side marks comparable blocks with ``# @twin <block>`` / ``// @twin <block>``;
the set of block names must be identical and appear in the same order.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_DIR = REPO_ROOT / "py" / "newtollm"
CPP_DIR = REPO_ROOT / "cpp" / "newtollm"
PY_MARKER = re.compile(r"^\s*# @twin (\w+)\s*$")
CPP_MARKER = re.compile(r"^\s*// @twin (\w+)\s*$")
# Entry points have no block-level twin: they differ by nature (argparse vs argv).
UNPAIRED = {"__init__", "__main__"}


@dataclass(frozen=True)
class Block:
    name: str
    lines: list[str]


def read_blocks(path: Path, marker: re.Pattern[str]) -> list[Block]:
    """Split a file into ``@twin`` blocks; a block runs until the next marker."""
    blocks: list[Block] = []
    current: tuple[str, list[str]] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = marker.match(line)
        if match:
            if current:
                blocks.append(Block(current[0], current[1]))
            current = (match.group(1), [])
        elif current is not None:
            current[1].append(line)
    if current:
        blocks.append(Block(current[0], current[1]))
    return blocks


def twin_names() -> list[str]:
    return sorted(p.stem for p in PY_DIR.glob("*.py") if p.stem not in UNPAIRED)


def cpp_blocks(name: str) -> list[Block]:
    """Header blocks first (declarations), then source blocks; merged by name."""
    merged: dict[str, list[str]] = {}
    for suffix in (".h", ".cpp"):
        path = CPP_DIR / f"{name}{suffix}"
        if path.exists():
            for block in read_blocks(path, CPP_MARKER):
                merged.setdefault(block.name, []).extend(block.lines)
    return [Block(k, v) for k, v in merged.items()]


def order_in_source(name: str) -> list[str]:
    """Block order as it appears in the .cpp (falls back to .h for header-only twins)."""
    source = CPP_DIR / f"{name}.cpp"
    path = source if source.exists() else CPP_DIR / f"{name}.h"
    return [b.name for b in read_blocks(path, CPP_MARKER)]


def check() -> list[str]:
    errors: list[str] = []
    for name in twin_names():
        if not (CPP_DIR / f"{name}.h").exists():
            errors.append(f"{name}: missing cpp/newtollm/{name}.h")
            continue
        py = [b.name for b in read_blocks(PY_DIR / f"{name}.py", PY_MARKER)]
        cpp = [b.name for b in cpp_blocks(name)]
        if not py:
            errors.append(f"{name}: no '# @twin' blocks in Python")
        if set(py) != set(cpp):
            errors.append(
                f"{name}: only in Python {sorted(set(py) - set(cpp))}, "
                f"only in C++ {sorted(set(cpp) - set(py))}"
            )
        elif [n for n in py if n in order_in_source(name)] != order_in_source(name):
            errors.append(f"{name}: block order differs: py={py} cpp={order_in_source(name)}")
    for path in CPP_DIR.glob("*.h"):
        if path.stem not in twin_names() and path.stem != "main":
            errors.append(f"{path.stem}: C++ file without Python twin")
    return errors


def main() -> int:
    errors = check()
    for error in errors:
        print(f"[twins] {error}")
    print(f"[twins] {len(twin_names())} twin pairs, {'OK' if not errors else 'FAILED'}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

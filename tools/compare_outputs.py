"""Compare two lesson transcripts: text must match exactly, numbers within a tolerance.

python tools/compare_outputs.py py.txt cpp.txt [--atol 1e-4]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

NUMBER = re.compile(r"-?\d+\.\d+|-?\d+")


@dataclass(frozen=True)
class Mismatch:
    line: int
    python: str
    cpp: str
    reason: str


def split_line(line: str) -> tuple[str, list[float]]:
    """Return the line with numbers replaced by '#' and the list of numbers."""
    numbers = [float(m) for m in NUMBER.findall(line)]
    return NUMBER.sub("#", line), numbers


def compare_lines(python: list[str], cpp: list[str], atol: float) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    if len(python) != len(cpp):
        mismatches.append(Mismatch(0, str(len(python)), str(len(cpp)), "different number of lines"))
    for i, (py_line, cpp_line) in enumerate(zip(python, cpp, strict=False), start=1):
        py_text, py_nums = split_line(py_line)
        cpp_text, cpp_nums = split_line(cpp_line)
        if py_text != cpp_text:
            mismatches.append(Mismatch(i, py_line, cpp_line, "text differs"))
        elif any(
            abs(a - b) > atol * max(1.0, abs(a)) for a, b in zip(py_nums, cpp_nums, strict=True)
        ):
            mismatches.append(Mismatch(i, py_line, cpp_line, f"numbers differ (atol={atol:g})"))
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python", type=Path)
    parser.add_argument("cpp", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    mismatches = compare_lines(
        args.python.read_text(errors="replace").splitlines(),
        args.cpp.read_text(errors="replace").splitlines(),
        args.atol,
    )
    for m in mismatches:
        print(f"line {m.line}: {m.reason}\n  py : {m.python}\n  cpp: {m.cpp}")
    print("IDENTICAL" if not mismatches else f"{len(mismatches)} mismatching line(s)")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())

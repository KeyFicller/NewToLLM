#!/usr/bin/env bash
# All static gates for TorchTwins. Usage: tools/lint.sh [build-dir]
# Needs: ruff, mypy, clang-format, run-clang-tidy and a configured build dir
# (cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON) for clang-tidy.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${1:-build}"
CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
RUN_CLANG_TIDY="${RUN_CLANG_TIDY:-run-clang-tidy}"
CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"

echo "== ruff ==";          ruff check py tools/compare_outputs.py tools/check_twins.py tools/parity.py tools/train_parity.py tools/bench.py tools/build_site.py tools/site_*.py tests/*.py
echo "== ruff format ==";   ruff format --check py tools/compare_outputs.py tools/check_twins.py tools/parity.py tools/train_parity.py tools/bench.py tools/build_site.py tools/site_*.py tests/*.py
echo "== mypy --strict =="; mypy
echo "== twins ==";         python3 tools/check_twins.py
echo "== clang-format =="; "$CLANG_FORMAT" --dry-run -Werror cpp/torchtwins/*.h cpp/torchtwins/*.cpp cpp/tools/*.cpp
echo "== clang-tidy ==";    "$RUN_CLANG_TIDY" -clang-tidy-binary "$CLANG_TIDY" -quiet -p "$BUILD_DIR" -j "$(nproc)" "$PWD/cpp/(torchtwins|tools)/.*\.cpp" > /tmp/clang-tidy.log 2>&1 \
  || { grep -E "(warning|error):" /tmp/clang-tidy.log | sort -u; exit 1; }
echo "lint OK"

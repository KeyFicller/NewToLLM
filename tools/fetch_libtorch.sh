#!/usr/bin/env bash
# Download the CPU LibTorch build that CI and the README use into vendor/libtorch.
#
#   tools/fetch_libtorch.sh              # version from tools/libtorch.version
#   tools/fetch_libtorch.sh 2.10.0       # override version
#   LIBTORCH_VENDOR_PATH=/other/path tools/fetch_libtorch.sh
#
# Re-running is a no-op when vendor/libtorch already looks complete, unless
# FORCE=1 is set.
set -euo pipefail

cd "$(dirname "$0")/.."
VERSION_FILE=tools/libtorch.version
VERSION="${1:-$(tr -d '[:space:]' < "$VERSION_FILE")}"
VENDOR="${LIBTORCH_VENDOR_PATH:-$PWD/vendor/libtorch}"
PARENT="$(dirname "$VENDOR")"
MARKER="$VENDOR/share/cmake/Torch/TorchConfig.cmake"
URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${VERSION}%2Bcpu.zip"

if [[ -f "$MARKER" && "${FORCE:-0}" != "1" ]]; then
  echo "LibTorch already present at $VENDOR (set FORCE=1 to re-download)."
  exit 0
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Fetching LibTorch ${VERSION} (CPU) -> $VENDOR"
curl -fL --retry 3 --retry-delay 2 -o "$tmpdir/libtorch.zip" "$URL"
mkdir -p "$PARENT"
rm -rf "$VENDOR"
unzip -q "$tmpdir/libtorch.zip" -d "$tmpdir/out"
# Upstream zip always contains a top-level "libtorch/" directory.
mv "$tmpdir/out/libtorch" "$VENDOR"

if [[ ! -f "$MARKER" ]]; then
  echo "error: download succeeded but $MARKER is missing" >&2
  exit 1
fi
echo "OK: $VENDOR (TorchConfig.cmake found)"

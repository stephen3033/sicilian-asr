#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This lock targets native macOS ARM64. See docs/dependencies.md. CUDA is not configured." >&2
  exit 1
fi
if ! command -v uv >/dev/null; then
  echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi
if ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "Install Apple's Command Line Tools with xcode-select --install, then rerun." >&2
  exit 1
fi

uv sync --frozen --extra research
.venv/bin/python scripts/download_resources.py --all
.venv/bin/python -m pytest
.venv/bin/python scripts/validate_resources.py
.venv/bin/python scripts/validate_environment.py --acoustic
echo "Initialization checks finished. Resource and device status: .cache/validation/"

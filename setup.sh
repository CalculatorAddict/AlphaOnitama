#!/usr/bin/env bash
# Phase 1 setup for macOS (Apple Silicon). Installs uv if needed, creates a
# virtualenv, and installs the package with dev tooling. Phase 2 (PyTorch) is
# opt-in via the [ml] extra; see the final echo.
set -e

if ! command -v brew &>/dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
if ! command -v uv &>/dev/null; then
  brew install uv
fi
uv venv .venv
uv pip install -e ".[dev]"      # Phase 1 + tooling

echo "Phase 1 ready. Run: uv run onitama --mode human-vs-mcts"
echo "For Phase 2: uv pip install -e '.[ml]'  then see train.py"

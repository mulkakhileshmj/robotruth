#!/usr/bin/env bash
# One-shot bootstrap of a fresh GPU box for robotruth. Runs ON the box.
# Usage (from local):  bash ops/sync_and_test.sh rt_main --co   (sync first), then
#   ssh box 'bash ~/rt_main/ops/bootstrap_box.sh'
# Starts, in parallel: dataset fetch (metadata + parquet only), torch install for the guard
# CUDA path, and the full test suite. Logs land in ~/logs.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/logs ~/datasets
cd ~/rt_main
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d .venv ] || uv venv .venv -q --python 3.12
uv pip install -q -e '.[dev]' opencv-python-headless huggingface_hub pyyaml --python .venv/bin/python
echo "deps ok $(date -u +%H:%M:%S)"
# 1. dataset fetch (background)
nohup .venv/bin/python ops/fetch_real_datasets.py ~/datasets > ~/logs/fetch.log 2>&1 &
echo "fetch started"
# 2. torch for the guard CUDA scorer (background, optional)
nohup uv pip install -q torch --index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python > ~/logs/torch.log 2>&1 &
echo "torch install started"
# 3. full test suite (foreground)
.venv/bin/python -m pytest -q 2>&1 | tail -5 | tee ~/logs/pytest.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "bootstrap done $(date -u +%H:%M:%S)"

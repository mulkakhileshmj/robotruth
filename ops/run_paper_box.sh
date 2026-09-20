#!/usr/bin/env bash
# Baseline comparison for the paper. Fetches the public corpora, then runs every detector.
# Usage: setsid nohup bash ops/run_paper_box.sh > ~/logs/paper.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/rt_main
export PATH=$HOME/.local/bin:$PATH
DATA=$HOME/real_data
VAL=$HOME/validation/paper
mkdir -p ~/logs "$DATA" "$VAL"

echo "== gpu health $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,ecc.errors.uncorrected.volatile.total --format=csv,noheader
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ -n "$ECC" ] && [ "$ECC" != "N/A" ] && [ "$ECC" != "0" ]; then
  echo "FATAL: GPU reports $ECC uncorrectable ECC errors"; exit 1
fi

echo "== deps $(date -u +%FT%TZ)"
.venv/bin/python -c "import sklearn, torch, huggingface_hub" 2>/dev/null || \
  uv pip install -q scikit-learn torch huggingface_hub --python .venv/bin/python > ~/logs/deps_paper.log 2>&1
echo "deps ready $(date -u +%FT%TZ)"

if [ ! -d "$DATA/kantine__BotFails" ]; then
  echo "== fetch $(date -u +%FT%TZ)"
  .venv/bin/python -u ops/fetch_real_datasets.py "$DATA" > ~/logs/paper_fetch.log 2>&1
fi

echo "== baselines $(date -u +%FT%TZ)"
.venv/bin/python -u ops/baseline_comparison.py "$DATA" "$VAL" --max-episodes 400
echo "PAPER_RUN_DONE $(date -u +%FT%TZ)"

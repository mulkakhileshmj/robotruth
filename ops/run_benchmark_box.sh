#!/usr/bin/env bash
# Runs on the GPU box, detached. Fetches the benchmark datasets (metadata and parquet only),
# validates every robotruth module over them, then renders the cross-dataset HTML report.
# Usage: setsid nohup bash ops/run_benchmark_box.sh > ~/logs/benchmark.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/rt_main
export PATH=$HOME/.local/bin:$PATH
DATA=$HOME/benchmark_data
VAL=$HOME/validation/benchmark
mkdir -p ~/logs "$DATA" "$VAL"
.venv/bin/python -c "import huggingface_hub" 2>/dev/null || uv pip install -q huggingface_hub --python .venv/bin/python
echo "== fetch $(date -u +%FT%TZ)"
nice -n 10 .venv/bin/python -u ops/benchmark_fetch.py "$DATA" --cap-gb 6 --per-family 10
.venv/bin/python -u ops/flatten_nested_datasets.py "$DATA"
echo "== validate $(date -u +%FT%TZ)"
nice -n 10 .venv/bin/python -u ops/validate_real.py "$DATA" "$VAL" --max-episodes 300
echo "== report $(date -u +%FT%TZ)"
.venv/bin/python -u ops/benchmark_report.py "$VAL"
echo "BENCHMARK_DONE $(date -u +%FT%TZ)"

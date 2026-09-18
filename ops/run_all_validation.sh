#!/usr/bin/env bash
# Run the full real-data validation suite on the box, in parallel where independent.
# Usage (on the box): bash ~/rt_main/ops/run_all_validation.sh
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/rt_main
PY=.venv/bin/python
OUT=~/validation
mkdir -p $OUT ~/logs
# LeRobot-format datasets (everything with meta/info.json), one runner over all
mkdir -p ~/datasets_lerobot
for info in $(find ~/datasets -name info.json -path "*/meta/*" | grep -v ".cache"); do
  d=$(dirname $(dirname "$info"))
  name=$(echo "$d" | sed "s|$HOME/datasets/||; s|/|__|g")
  ln -sfn "$d" ~/datasets_lerobot/"$name"
done
ls ~/datasets_lerobot
$PY ops/validate_real.py ~/datasets_lerobot $OUT/lerobot --max-episodes 400 > ~/logs/validate.log 2>&1 &
V1=$!
# RoboArena pairwise analysis
if [ -d ~/datasets/RoboArena__DataDump_02-03-2026 ]; then
  $PY ops/roboarena_pairs.py ~/datasets/RoboArena__DataDump_02-03-2026 $OUT/roboarena > ~/logs/roboarena.log 2>&1 &
  V2=$!
fi
# SO101 eval CSV -> claims audit + compare
if [ -d ~/datasets/cons909__V1-LeRobot-SO101-Eval-Videos ]; then
  ($PY ops/so101_eval_convert.py ~/datasets/cons909__V1-LeRobot-SO101-Eval-Videos $OUT/so101_eval     && $PY -m robotruth.cli stats audit $OUT/so101_eval/claims.csv --out-dir $OUT/so101_eval > ~/logs/so101.log 2>&1) &
fi
wait ${V1:-} ${V2:-} 2>/dev/null
echo "=== validate.log tail ==="; tail -12 ~/logs/validate.log
echo "=== roboarena.log tail ==="; tail -6 ~/logs/roboarena.log 2>/dev/null
echo "validation done $(date -u +%H:%M:%S)"

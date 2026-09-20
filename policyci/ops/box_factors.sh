#!/usr/bin/env bash
# Runs ON the GPU box. Second Policy CI experiment: named scene factors, real failure
# regions, and a noise floor measured against a source of variance that actually exists.
#
# The first sweep could only say "74 scenarios broke". This one asks three questions the
# first could not:
#
#   1. Does the policy have a real weak region, and does clustering find it?
#      The battery varies cube_x, cube_y and cube_yaw_deg. gym-aloha NEVER rotates the cube,
#      in training or evaluation, so yaw is genuinely out of distribution. If the policy is
#      brittle to rotation, the hotspot finder should name it without being told.
#
#   2. Is the cell deterministic, and does the tool say so rather than implying a gap?
#      act_v18 twice at different policy seeds.
#
#   3. Does the noise floor work when there IS noise?
#      act_noise runs the same weights with Gaussian action noise, twice. That is a
#      stochastic source without being a degradation: unchanged in expectation, it simply
#      stops repeating itself exactly.
#
# Usage: bash box_factors.sh <work_dir> <N> <SHARDS> <NOISE_SIGMA>
set -uo pipefail
WORK="${1:?work dir}"
N="${2:-256}"
SHARDS="${3:-6}"
SIGMA="${4:-0.01}"

export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$WORK/robotruth"

if ! dpkg -s libegl1 >/dev/null 2>&1; then
  sudo apt-get update -qq >/dev/null 2>&1
  sudo apt-get install -y -qq libegl1 libosmesa6 ffmpeg >/dev/null 2>&1
fi
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -d .venv ] || uv venv .venv -q --python 3.12
uv pip install -q -e '.[dev]' --python .venv/bin/python 2>&1 | tail -2
uv pip install -q -e 'policyci[aloha,dev]' opencv-python-headless --python .venv/bin/python 2>&1 | tail -2
PY=.venv/bin/python
CLI=.venv/bin/policyci

echo "=== tests on $(hostname) ==="
if ! $PY -m pytest policyci/tests -q 2>&1 | tail -4; then
  echo "ABORT: tests failed"; exit 1
fi

OUT="$WORK/out"; mkdir -p "$OUT/logs"
echo "=== factor battery ==="
$CLI battery --factors --n "$N" --base-seed 0 -o "$OUT/battery_f1.jsonl"

echo "=== smoke: 2 scenarios through the factor path ==="
$CLI battery --factors --n 2 --base-seed 999 -o "$OUT/battery_smoke.jsonl" >/dev/null
if ! $CLI run --battery "$OUT/battery_smoke.jsonl" --policy-name smoke \
      --out "$OUT/smoke" --progress-every 1 2>&1 | tail -3; then
  echo "ABORT: smoke failed"; exit 1
fi
[ -s "$OUT/smoke/episodes.jsonl" ] || { echo "ABORT: smoke wrote nothing"; exit 1; }
echo "smoke ok"

$PY -c "from huggingface_hub import snapshot_download as d; d('lerobot/act_aloha_sim_transfer_cube_human')" >/dev/null 2>&1

# name|bias|seed|render|action_noise
RUNS="act_v18|0|0|1 act_v18_s1|0|1|0 act_noise_a|0|0|0 act_noise_b|0|1|0 act_v19_bias005|0.05|0|1"

echo "=== launching $(echo $RUNS | wc -w) runs x $SHARDS shards on $(nproc --all) vCPUs ==="
PIDS=""
for R in $RUNS; do
  NAME=$(echo "$R" | cut -d'|' -f1)
  BIAS=$(echo "$R" | cut -d'|' -f2)
  SEED=$(echo "$R" | cut -d'|' -f3)
  REND=$(echo "$R" | cut -d'|' -f4)
  for K in $(seq 0 $((SHARDS - 1))); do
    ARGS="--battery $OUT/battery_f1.jsonl --policy-name $NAME --run-id $NAME"
    ARGS="$ARGS --out $OUT/${NAME}/shard${K} --policy-seed $SEED --shard $K --num-shards $SHARDS --progress-every 10"
    [ "$BIAS" != "0" ] && ARGS="$ARGS --state-bias $BIAS"
    [ "$REND" = "1" ] && ARGS="$ARGS --render"
    case "$NAME" in act_noise_*) ARGS="$ARGS --action-noise $SIGMA";; esac
    $CLI run $ARGS > "$OUT/logs/${NAME}_s${K}.log" 2>&1 &
    PIDS="$PIDS $!"
  done
done
echo "pids:$PIDS"

while :; do
  ALIVE=0; for P in $PIDS; do kill -0 "$P" 2>/dev/null && ALIVE=1; done
  [ "$ALIVE" -eq 0 ] && break
  sleep 120
  echo "[$(date -u +%H:%M:%S)] episodes: $(cat "$OUT"/*/shard*/episodes.jsonl 2>/dev/null | wc -l)"
done
for P in $PIDS; do wait "$P" 2>/dev/null; done
echo "=== all shards finished ==="

for R in $RUNS; do
  NAME=$(echo "$R" | cut -d'|' -f1)
  $CLI merge "$OUT/$NAME"/shard*/run_manifest.json --battery "$OUT/battery_f1.jsonl" \
    --out "$OUT/merged/$NAME" || echo "MERGE FAILED $NAME"
  mkdir -p "$OUT/merged/$NAME/videos"
  cp "$OUT/$NAME"/shard*/videos/*.mp4 "$OUT/merged/$NAME/videos/" 2>/dev/null
done

A="$OUT/merged/act_v18/run_manifest.json"
A2="$OUT/merged/act_v18_s1/run_manifest.json"
NA="$OUT/merged/act_noise_a/run_manifest.json"
NB="$OUT/merged/act_noise_b/run_manifest.json"
BAT="$OUT/battery_f1.jsonl"

echo "=== Q2: is the deterministic cell detected and labelled? ==="
$CLI diff "$A" "$A2" --battery "$BAT" -o "$OUT/diff_determinism.md"

echo "=== Q3: does the floor work when there IS noise? (same weights, action noise) ==="
$CLI diff "$NA" "$NB" --battery "$BAT" -o "$OUT/diff_noise_pair.md"

echo "=== Q1: the real regression, with hotspots and a MEASURED floor ==="
$CLI diff "$A" "$OUT/merged/act_v19_bias005/run_manifest.json" --noise "$NA" "$NB" \
  --battery "$BAT" -o "$OUT/diff_bias005.md" --html "$OUT/browser_bias005.html"
echo "bias exit: $?"

echo "=== Q1b: where is the BASELINE itself weak? (its own failures, clustered) ==="
$PY - "$A" "$BAT" "$OUT/baseline_weakness.md" <<'PYEOF'
import json, sys
from pathlib import Path
from policyci.cluster import find_hotspots
from policyci.scenario import Battery
m = json.loads(Path(sys.argv[1]).read_text())
bat = Battery.load(sys.argv[2])
params = {s.hash: s.params for s in bat.scenarios}
failed = {h for h, r in m["results"].items() if not r["success"]}
spots = find_hotspots(params, failed, list(params))
lines = ["# Where the baseline itself is weak", "",
         f"{len(failed)} of {len(m['results'])} scenarios failed under act_v18 "
         f"(no candidate involved: this is the policy's own envelope).", ""]
if spots:
    lines += ["| region | inside | elsewhere | lift | p |", "|---|---|---|---|---|"]
    for h in spots:
        lines.append(f"| `{h.description}` | {h.n_broken_inside}/{h.n_inside} "
                     f"({100*h.inside_rate.estimate:.1f}% [{100*h.inside_rate.lower:.0f}, "
                     f"{100*h.inside_rate.upper:.0f}]) | {h.n_broken_outside}/{h.n_outside} "
                     f"({100*h.outside_rate.estimate:.1f}%) | {h.lift:.1f}x | {h.p_value:.1e} |")
        print(f"  {h}")
else:
    lines.append("_No region concentrates the baseline's failures beyond chance._")
    print("  no region found")
Path(sys.argv[3]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PYEOF

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt" 2>/dev/null
{ echo "host: $(hostname)"; echo "date_utc: $(date -u)"; echo "vcpus: $(nproc --all)"; } > "$OUT/box.txt"
tar -czf "$WORK/policyci_out.tgz" -C "$OUT" battery_f1.jsonl logs merged \
  $(cd "$OUT" && ls diff_*.md browser_*.html baseline_weakness.md gpu.txt box.txt 2>/dev/null) 2>/dev/null
echo "DONE: pull $WORK/policyci_out.tgz ($(du -h "$WORK/policyci_out.tgz" | cut -f1))"

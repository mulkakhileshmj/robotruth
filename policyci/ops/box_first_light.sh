#!/usr/bin/env bash
# Runs ON the GPU box. Driven by policyci/ops/run_policyci_box.sh, which copies this file
# over and executes it, so nothing depends on shell quoting through ssh.
#
# The experiment:
#   A   = act_v18, policy_seed 0            the baseline
#   A2  = act_v18, policy_seed 1            the A-vs-A noise floor, identical scenes
#   B_k = act_v19_bias<k>, policy_seed 0    a dose-response ladder of controlled
#                                           regressions (proprioception bias; weights
#                                           are never touched)
# Every run is sharded across workers purely for wall clock: scenario identity, the
# per-scenario policy seed and the evaluator all derive from the scenario hash, so shards
# produce exactly what one process would, and the merge refuses to emit a manifest unless
# every scenario is covered exactly once.
#
# Usage: bash box_first_light.sh <work_dir> <N> <SHARDS> "<BIASES>"
set -uo pipefail
WORK="${1:?work dir}"
N="${2:-200}"
SHARDS="${3:-4}"
BIASES="${4:-0.02 0.05 0.10}"

export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl
# One worker per core beats many threads per worker. With N workers already running,
# per-process BLAS/OMP threading only oversubscribes: the first sweep hit load average
# 123 on 30 cores and spent the difference in context switches.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "$WORK/robotruth"

# MuJoCo needs an EGL vendor library to render offscreen; a stock Lambda image has none.
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

echo "=== policyci tests on $(hostname) ==="
if ! $PY -m pytest policyci/tests -q 2>&1 | tail -5; then
  echo "ABORT: tests failed, not spending GPU time on a broken tree"; exit 1
fi

OUT="$WORK/out"
mkdir -p "$OUT/logs"
echo "=== battery ==="
$CLI battery --n "$N" --base-seed 0 -o "$OUT/battery_b1.jsonl"

# Smoke test: one real episode end to end before committing the box to a long sweep.
# Unit tests cannot catch a runtime failure in the simulator, policy or record path;
# this does, in about a minute, instead of after every worker has burned an hour.
echo "=== smoke test: 2 scenarios, full path ==="
$CLI battery --n 2 --base-seed 999 -o "$OUT/battery_smoke.jsonl" >/dev/null
if ! $CLI run --battery "$OUT/battery_smoke.jsonl" --policy-name smoke \
      --out "$OUT/smoke" --policy-seed 0 --render --progress-every 1 2>&1 | tail -4; then
  echo "ABORT: smoke test failed, see above"; exit 1
fi
if [ ! -s "$OUT/smoke/episodes.jsonl" ]; then
  echo "ABORT: smoke test wrote no episode records"; exit 1
fi
echo "smoke ok: $(wc -l < "$OUT/smoke/episodes.jsonl") episode records written"

echo "=== warm the checkpoint cache once, before the parallel readers start ==="
$PY -c "from huggingface_hub import snapshot_download as d; d('lerobot/act_aloha_sim_transfer_cube_human')" >/dev/null 2>&1

# name|bias|seed|render
RUNS="act_v18|0|0|1 act_v18_s1|0|1|0"
for BZ in $BIASES; do
  TAG=$(echo "$BZ" | tr -d '.')
  RUNS="$RUNS act_v19_bias${TAG}|${BZ}|0|1"
done

echo "=== launching: $(echo $RUNS | wc -w) runs x $SHARDS shards on $(nproc --all) vCPUs ==="
PIDS=""
for R in $RUNS; do
  NAME=$(echo "$R" | cut -d'|' -f1)
  BIAS=$(echo "$R" | cut -d'|' -f2)
  SEED=$(echo "$R" | cut -d'|' -f3)
  REND=$(echo "$R" | cut -d'|' -f4)
  for K in $(seq 0 $((SHARDS - 1))); do
    ARGS="--battery $OUT/battery_b1.jsonl --policy-name $NAME --run-id $NAME"
    ARGS="$ARGS --out $OUT/${NAME}/shard${K} --policy-seed $SEED --shard $K --num-shards $SHARDS"
    ARGS="$ARGS --progress-every 10"
    [ "$BIAS" != "0" ] && ARGS="$ARGS --state-bias $BIAS"
    [ "$REND" = "1" ] && ARGS="$ARGS --render"
    $CLI run $ARGS > "$OUT/logs/${NAME}_s${K}.log" 2>&1 &
    PIDS="$PIDS $!"
  done
done
echo "pids:$PIDS"

while :; do
  ALIVE=0
  for P in $PIDS; do kill -0 "$P" 2>/dev/null && ALIVE=1; done
  [ "$ALIVE" -eq 0 ] && break
  sleep 120
  DONE_EP=$(cat "$OUT"/*/shard*/episodes.jsonl 2>/dev/null | wc -l)
  echo "[$(date -u +%H:%M:%S)] episodes done: $DONE_EP  | $(tail -n1 "$OUT/logs/act_v18_s0.log" 2>/dev/null)"
done
for P in $PIDS; do wait "$P" 2>/dev/null; done
echo "=== all shards finished ==="

echo "=== merging shards ==="
MERGED=""
for R in $RUNS; do
  NAME=$(echo "$R" | cut -d'|' -f1)
  $CLI merge "$OUT/$NAME"/shard*/run_manifest.json --battery "$OUT/battery_b1.jsonl" \
    --out "$OUT/merged/$NAME" || { echo "MERGE FAILED for $NAME"; continue; }
  mkdir -p "$OUT/merged/$NAME/videos"
  cp "$OUT/$NAME"/shard*/videos/*.mp4 "$OUT/merged/$NAME/videos/" 2>/dev/null
  MERGED="$MERGED $NAME"
done
echo "merged:$MERGED"

A="$OUT/merged/act_v18/run_manifest.json"
A2="$OUT/merged/act_v18_s1/run_manifest.json"
echo "=== NEGATIVE CONTROL: A vs A2, same policy, different policy seed."
echo "=== The engine must NOT call this a regression."
$CLI diff "$A" "$A2" -o "$OUT/diff_noise_control.md"
echo "control exit: $?"

for BZ in $BIASES; do
  TAG=$(echo "$BZ" | tr -d '.')
  echo "=== DIFF: bias $BZ vs baseline, against the measured noise floor ==="
  $CLI diff "$A" "$OUT/merged/act_v19_bias${TAG}/run_manifest.json" --noise "$A" "$A2" \
    -o "$OUT/diff_bias${TAG}.md"
  echo "bias $BZ exit: $?"
done

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt" 2>/dev/null
{ echo "host: $(hostname)"; echo "date_utc: $(date -u)"; echo "vcpus: $(nproc)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$OUT/box.txt" 2>/dev/null
du -sh "$OUT"/merged/*/videos 2>/dev/null

tar -czf "$WORK/policyci_out.tgz" -C "$OUT" \
  battery_b1.jsonl logs merged diff_noise_control.md $(cd "$OUT" && ls diff_bias*.md 2>/dev/null) gpu.txt box.txt 2>/dev/null
echo "DONE: pull $WORK/policyci_out.tgz ($(du -h "$WORK/policyci_out.tgz" 2>/dev/null | cut -f1))"

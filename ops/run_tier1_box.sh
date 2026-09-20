#!/usr/bin/env bash
# Tier 1 production-readiness experiments, run detached on the GPU box and in parallel.
# Usage: setsid nohup bash ops/run_tier1_box.sh > ~/logs/tier1.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/rt_main
export PATH=$HOME/.local/bin:$PATH
export MUJOCO_GL=egl
VAL=$HOME/validation/tier1
DATA=$HOME/real_data
EPS=$HOME/rt_main/examples/validation/2026-09-19/guard_detection_v2/multi_head/episodes
mkdir -p ~/logs "$VAL" "$DATA"

# A rented GPU can be physically faulty. One box burned an hour of rollouts before its
# uncorrectable DRAM ECC count (64) surfaced in the job logs as CUDA errors, and every
# number it produced had to be thrown away. Check the card before trusting anything on it.
echo "== gpu health $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,ecc.errors.uncorrected.volatile.total --format=csv,noheader
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null | tr -d ' ')
if [ -n "$ECC" ] && [ "$ECC" != "N/A" ] && [ "$ECC" != "0" ]; then
  echo "FATAL: GPU reports $ECC uncorrectable ECC errors. Results from this card cannot be trusted."
  exit 1
fi
echo "== deps $(date -u +%FT%TZ)"
if ! ldconfig -p | grep -q libEGL.so.1; then
  sudo apt-get update -qq > ~/logs/apt.log 2>&1
  sudo apt-get install -y libegl1 libosmesa6 ffmpeg >> ~/logs/apt.log 2>&1
fi
.venv/bin/python -c "import lerobot, gym_aloha" 2>/dev/null || \
  uv pip install -q lerobot gym-aloha torch --python .venv/bin/python > ~/logs/deps_lerobot.log 2>&1
.venv/bin/python -c "import gym_pusht, diffusers" 2>/dev/null || \
  uv pip install -q gym-pusht diffusers --python .venv/bin/python > ~/logs/deps_pusht.log 2>&1
.venv/bin/python -c "import huggingface_hub" 2>/dev/null || \
  uv pip install -q huggingface_hub --python .venv/bin/python
echo "deps ready $(date -u +%FT%TZ)"

# Arithmetic self-check, now that torch exists. A card can pass its ECC counter and still
# return garbage, and a silent numerical fault would look like a bad experimental result.
.venv/bin/python - <<'PY' || { echo "FATAL: GPU arithmetic self-check failed"; exit 1; }
import torch
assert torch.cuda.is_available(), "no CUDA device"
a = torch.randn(4096, 4096, device="cuda")
assert torch.isfinite(a @ a.T).all(), "non-finite result from a clean matmul"
print("gpu self-check ok:", torch.cuda.get_device_name(0))
PY

if [ ! -d "$EPS" ]; then
  echo "FATAL: saved calibration episodes not found at $EPS"; exit 1
fi

echo "== launching experiments in parallel $(date -u +%FT%TZ)"
setsid nohup .venv/bin/python -u ops/guard_experiments.py fa "$VAL" "$EPS" 400 \
  > ~/logs/t1_fa.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python -u ops/guard_experiments.py ramp "$VAL" "$EPS" 10 \
  > ~/logs/t1_ramp.log 2>&1 < /dev/null &
setsid nohup .venv/bin/python -u ops/guard_experiments.py combo "$VAL" act_aloha_insertion 200 20 10 \
  > ~/logs/t1_insertion.log 2>&1 < /dev/null &
# diffusion_pusht is deliberately not launched here. Loading that checkpoint onto an A10
# raises "CUDA error: uncorrectable ECC error" while copying diffusion.unet.final_conv, and
# it poisons the GPU for every other process on the box: two separate A10s went from 0 to
# exactly 64 uncorrectable DRAM errors within a minute of that load, taking the other jobs
# with them, and a GPU reset cleared the counter both times. Identical counts on different
# cards mean this is triggered by the load, not by defective memory. Run it alone, on a box
# nothing else is using, if you want to chase it.

# real logs needs the public datasets first; it is CPU work and runs alongside the sims
( .venv/bin/python -u ops/fetch_real_datasets.py "$DATA" > ~/logs/t1_fetch.log 2>&1
  nice -n 10 .venv/bin/python -u ops/guard_real_logs.py "$DATA" "$VAL" --max-episodes 400 \
    > ~/logs/t1_reallogs.log 2>&1 ) &

echo "launched: fa, ramp, insertion, pusht, real_logs"
wait
echo "TIER1_ALL_DONE $(date -u +%FT%TZ)"

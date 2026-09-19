#!/usr/bin/env bash
# Runs on the GPU box, detached (setsid), so it survives the SSH session ending.
# Installs the live-loop deps once, then runs the fault-injection guard experiment.
# Usage: setsid nohup bash ops/run_guard_detection_box.sh <out_dir> <n_cal> <n_nom> <n_fault> > ~/logs/guard_detection.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/rt_main
export PATH=$HOME/.local/bin:$PATH
mkdir -p ~/logs
OUT="${1:-$HOME/validation/guard_detection}"; N_CAL="${2:-200}"; N_NOM="${3:-20}"; N_FAULT="${4:-10}"
if ! ldconfig -p | grep -q libEGL.so.1; then
  sudo apt-get update -qq > ~/logs/apt.log 2>&1
  sudo apt-get install -y libegl1 libosmesa6 >> ~/logs/apt.log 2>&1
fi
if ! .venv/bin/python -c "import lerobot, gym_aloha" 2>/dev/null; then
  uv pip install -q lerobot gym-aloha torch --python .venv/bin/python > ~/logs/lerobot_install.log 2>&1
  echo install_done >> ~/logs/lerobot_install.log
fi
echo "deps ready $(date -u +%FT%TZ)"
MUJOCO_GL=egl .venv/bin/python -u ops/guard_detection.py "$OUT" "$N_CAL" "$N_NOM" "$N_FAULT"
echo "exit $? $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# Policy CI first-light on the Lambda box. Nothing runs on the local CPU.
#
# What it does THERE:
#   1. sync the robotruth tree (including policyci/), install both packages + aloha extras
#   2. run the policyci test suite
#   3. sample battery B1 (N scenarios)
#   4. run A, A2 and B IN PARALLEL on the one GPU (ACT is ~100 MB; three MuJoCo envs fit):
#        A  = act_v18            policy_seed 0    the baseline
#        A2 = act_v18            policy_seed 1    the A-vs-A noise floor
#        B  = act_v19_biased     policy_seed 0    the controlled regression
#   5. diff B vs A with the (A, A2) noise floor, plus the noise-only diff
#   6. tar the run outputs for pull-back
#
# Usage: bash policyci/ops/run_policyci_box.sh <remote_dir> [N] [STATE_BIAS]
#   ROBOTRUTH_HOST=ubuntu@<ip> bash policyci/ops/run_policyci_box.sh pci_1 200 0.06
set -euo pipefail
KEY="${ROBOTRUTH_SSH_KEY:-/c/Users/Lenovo/.ssh/id_ed25519_annotate_vast}"
HOST="${ROBOTRUTH_HOST:?set ROBOTRUTH_HOST=ubuntu@<box-ip>}"
REMOTE_DIR="${1:-pci_1}"
N="${2:-200}"
BIAS="${3:-0.06}"
TGZ="$(mktemp -u /c/Users/Lenovo/AppData/Local/Temp/robotruth_sync_XXXXXX).tgz"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $HOST"

cd /d && tar --exclude=robotruth/.venv --exclude=robotruth/.git --exclude=robotruth/.env \
  --exclude='robotruth/examples' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.ruff_cache' --exclude='robotruth/dist' -czf "$TGZ" robotruth
scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new "$TGZ" "$HOST:~/${REMOTE_DIR}.tgz"
rm -f "$TGZ"

$SSH "export PATH=\$HOME/.local/bin:\$PATH; set -e
  mkdir -p ~/$REMOTE_DIR && cd ~/$REMOTE_DIR
  rm -rf robotruth 2>/dev/null || true
  tar -xzf ~/${REMOTE_DIR}.tgz -C . && rm -f ~/${REMOTE_DIR}.tgz
  cd robotruth
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -d .venv ] || uv venv .venv -q --python 3.12
  uv pip install -q -e '.[dev]' --python .venv/bin/python 2>&1 | tail -2
  uv pip install -q -e 'policyci[aloha,dev]' opencv-python-headless --python .venv/bin/python 2>&1 | tail -2
  export MUJOCO_GL=egl
  echo '=== policyci tests on' \$(hostname) '==='
  .venv/bin/python -m pytest policyci/tests -q 2>&1 | tail -10
  OUT=~/$REMOTE_DIR/out && mkdir -p \$OUT/logs
  echo '=== battery ==='
  .venv/bin/policyci battery --n $N --base-seed 0 -o \$OUT/battery_b1.jsonl
  echo '=== warming the checkpoint cache (one download, three readers) ==='
  .venv/bin/python -c \"from huggingface_hub import snapshot_download as d; d('lerobot/act_aloha_sim_transfer_cube_human')\" >/dev/null 2>&1
  echo '=== runs A, A2, B in parallel ==='
  ( .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v18 \
      --out \$OUT/run_a --policy-seed 0 --render --progress-every 25 > \$OUT/logs/run_a.log 2>&1 ) &
  PA=\$!
  ( .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v18 --run-id act_v18_s1 \
      --out \$OUT/run_a2 --policy-seed 1 --progress-every 25 > \$OUT/logs/run_a2.log 2>&1 ) &
  PA2=\$!
  ( .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v19_biased \
      --state-bias $BIAS --out \$OUT/run_b --policy-seed 0 --render --progress-every 25 > \$OUT/logs/run_b.log 2>&1 ) &
  PB=\$!
  while kill -0 \$PA 2>/dev/null || kill -0 \$PA2 2>/dev/null || kill -0 \$PB 2>/dev/null; do
    sleep 60
    echo \"[\$(date -u +%H:%M:%S)] \$(tail -n1 \$OUT/logs/run_a.log 2>/dev/null) | \$(tail -n1 \$OUT/logs/run_a2.log 2>/dev/null) | \$(tail -n1 \$OUT/logs/run_b.log 2>/dev/null)\"
  done
  wait \$PA; wait \$PA2; wait \$PB
  tail -3 \$OUT/logs/run_a.log \$OUT/logs/run_a2.log \$OUT/logs/run_b.log
  echo '=== diff: B vs A, with the A-vs-A noise floor ==='
  .venv/bin/policyci diff \$OUT/run_a/run_manifest.json \$OUT/run_b/run_manifest.json \
    --noise \$OUT/run_a/run_manifest.json \$OUT/run_a2/run_manifest.json \
    -o \$OUT/diff_v19_vs_v18.md || true
  echo '=== diff: A vs A2 (the noise floor itself; must NOT be called a regression) ==='
  .venv/bin/policyci diff \$OUT/run_a/run_manifest.json \$OUT/run_a2/run_manifest.json \
    -o \$OUT/diff_noise_only.md || true
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > \$OUT/gpu.txt || true
  tar -czf ~/$REMOTE_DIR/policyci_out.tgz -C \$OUT .
  echo 'DONE: pull ~/'$REMOTE_DIR'/policyci_out.tgz'"

#!/usr/bin/env bash
# Policy CI first-light on the Lambda box. Nothing runs on the local CPU.
#
# What it does THERE:
#   1. sync the robotruth tree (including policyci/), install both packages + aloha extras
#   2. run the policyci test suite
#   3. sample battery B1 (N scenarios)
#   4. run A  = act_v18 (real checkpoint, normalization attached), policy_seed 0
#   5. run A2 = same policy, policy_seed 1            -> noise floor
#   6. run B  = act_v19_biased (state_bias)           -> the controlled regression
#   7. diff B vs A with the (A, A2) noise floor, render the report
#   8. tar the run outputs for pull-back
#
# Usage: bash policyci/ops/run_policyci_box.sh <remote_dir> [N] [STATE_BIAS]
#   bash policyci/ops/run_policyci_box.sh pci_1 200 0.06
# Then pull:
#   scp -i $KEY $HOST:~/pci_1/policyci_out.tgz policyci/examples/validation/<date>/
set -euo pipefail
KEY="${ROBOTRUTH_SSH_KEY:-/c/Users/Lenovo/.ssh/id_ed25519_annotate_vast}"
HOST="${ROBOTRUTH_HOST:?set ROBOTRUTH_HOST=ubuntu@<box-ip>}"
REMOTE_DIR="${1:-pci_1}"
N="${2:-200}"
BIAS="${3:-0.06}"
LOCAL_ROOT="/d/robotruth"
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
  mkdir -p robotruth && tar -xzf ~/${REMOTE_DIR}.tgz -C . && rm -f ~/${REMOTE_DIR}.tgz
  cd robotruth
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -d .venv ] || uv venv .venv -q --python 3.12
  uv pip install -q -e '.[dev]' --python .venv/bin/python 2>&1 | tail -1
  uv pip install -q -e 'policyci[aloha,dev]' opencv-python-headless --python .venv/bin/python 2>&1 | tail -1
  export MUJOCO_GL=egl
  echo '=== policyci tests on' \$(hostname) '==='
  .venv/bin/python -m pytest policyci/tests -q 2>&1 | tail -10
  OUT=~/$REMOTE_DIR/out && mkdir -p \$OUT
  echo '=== battery ==='
  .venv/bin/policyci battery --n $N --base-seed 0 -o \$OUT/battery_b1.jsonl
  echo '=== run A (act_v18, seed 0) ==='
  .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v18 \
    --out \$OUT/run_a --policy-seed 0 --render
  echo '=== run A2 (act_v18, seed 1: noise floor) ==='
  .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v18 --run-id act_v18_s1 \
    --out \$OUT/run_a2 --policy-seed 1
  echo '=== run B (act_v19_biased, state_bias $BIAS) ==='
  .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v19_biased \
    --state-bias $BIAS --out \$OUT/run_b --policy-seed 0 --render
  echo '=== diff ==='
  .venv/bin/policyci diff \$OUT/run_a/run_manifest.json \$OUT/run_b/run_manifest.json \
    --noise \$OUT/run_a/run_manifest.json \$OUT/run_a2/run_manifest.json \
    -o \$OUT/diff_v19_vs_v18.md || true
  .venv/bin/policyci diff \$OUT/run_a/run_manifest.json \$OUT/run_a2/run_manifest.json \
    -o \$OUT/diff_noise_only.md || true
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > \$OUT/gpu.txt || true
  tar -czf ~/$REMOTE_DIR/policyci_out.tgz -C \$OUT .
  echo 'DONE: pull ~/'$REMOTE_DIR'/policyci_out.tgz'"

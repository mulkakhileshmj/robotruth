#!/usr/bin/env bash
# Policy CI first light on a Lambda box. Nothing runs on the local CPU.
#
# The experiment:
#   A   = act_v18, policy_seed 0                 the baseline
#   A2  = act_v18, policy_seed 1                 the A-vs-A noise floor (same scenes)
#   B_k = act_v19_bias<k>, policy_seed 0         a dose-response ladder of controlled
#                                                regressions (proprioception bias, weights
#                                                never touched)
# All of them run in parallel on the one GPU: ACT is ~100 MB and MuJoCo is CPU-bound, so
# the box's 30 vCPUs are the real resource. Then every B is diffed against A with the
# (A, A2) noise floor, and A is diffed against A2 as the negative control: the engine must
# NOT call that a regression.
#
# Usage: ROBOTRUTH_HOST=ubuntu@<ip> bash policyci/ops/run_policyci_box.sh <dir> [N] [BIASES]
#   ROBOTRUTH_HOST=ubuntu@1.2.3.4 bash policyci/ops/run_policyci_box.sh pci_1 200 "0.02 0.05 0.10"
set -euo pipefail
KEY="${ROBOTRUTH_SSH_KEY:-/c/Users/Lenovo/.ssh/id_ed25519_annotate_vast}"
HOST="${ROBOTRUTH_HOST:?set ROBOTRUTH_HOST=ubuntu@<box-ip>}"
REMOTE_DIR="${1:-pci_1}"
N="${2:-200}"
BIASES="${3:-0.02 0.05 0.10}"
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
  .venv/bin/python -m pytest policyci/tests -q 2>&1 | tail -8
  OUT=~/$REMOTE_DIR/out && mkdir -p \$OUT/logs
  echo '=== battery ==='
  .venv/bin/policyci battery --n $N --base-seed 0 -o \$OUT/battery_b1.jsonl
  echo '=== warm the checkpoint cache once, before the parallel readers start ==='
  .venv/bin/python -c \"from huggingface_hub import snapshot_download as d; d('lerobot/act_aloha_sim_transfer_cube_human')\" >/dev/null 2>&1
  echo '=== launching runs in parallel ==='
  PIDS=''
  .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v18 \
      --out \$OUT/run_a --policy-seed 0 --render > \$OUT/logs/run_a.log 2>&1 &
  PIDS=\"\$PIDS \$!\"
  .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v18 --run-id act_v18_s1 \
      --out \$OUT/run_a2 --policy-seed 1 > \$OUT/logs/run_a2.log 2>&1 &
  PIDS=\"\$PIDS \$!\"
  for BZ in $BIASES; do
    TAG=\$(echo \$BZ | tr -d '.')
    .venv/bin/policyci run --battery \$OUT/battery_b1.jsonl --policy-name act_v19_bias\$TAG \
        --state-bias \$BZ --out \$OUT/run_b\$TAG --policy-seed 0 --render > \$OUT/logs/run_b\$TAG.log 2>&1 &
    PIDS=\"\$PIDS \$!\"
  done
  echo \"pids:\$PIDS\"
  while :; do
    ALIVE=0
    for P in \$PIDS; do kill -0 \$P 2>/dev/null && ALIVE=1; done
    [ \$ALIVE -eq 0 ] && break
    sleep 120
    echo \"[\$(date -u +%H:%M:%S)]\"; for L in \$OUT/logs/*.log; do echo \"  \$(basename \$L): \$(tail -n1 \$L)\"; done
  done
  for P in \$PIDS; do wait \$P || true; done
  echo '=== all runs finished ==='
  for L in \$OUT/logs/*.log; do echo \"--- \$(basename \$L)\"; tail -n2 \$L; done
  echo '=== NEGATIVE CONTROL: A vs A2, same policy. Must not be called a regression. ==='
  .venv/bin/policyci diff \$OUT/run_a/run_manifest.json \$OUT/run_a2/run_manifest.json \
    -o \$OUT/diff_noise_control.md || true
  for BZ in $BIASES; do
    TAG=\$(echo \$BZ | tr -d '.')
    echo \"=== DIFF: bias \$BZ vs baseline, with the measured noise floor ===\"
    .venv/bin/policyci diff \$OUT/run_a/run_manifest.json \$OUT/run_b\$TAG/run_manifest.json \
      --noise \$OUT/run_a/run_manifest.json \$OUT/run_a2/run_manifest.json \
      -o \$OUT/diff_bias\$TAG.md || true
  done
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > \$OUT/gpu.txt || true
  du -sh \$OUT/*/videos 2>/dev/null || true
  tar -czf ~/$REMOTE_DIR/policyci_out.tgz -C \$OUT .
  echo 'DONE: pull ~/'$REMOTE_DIR'/policyci_out.tgz'"

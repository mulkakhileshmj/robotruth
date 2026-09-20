#!/usr/bin/env bash
# Drive Policy CI first light on a rented GPU box. Nothing runs on the local CPU: this
# script only syncs the tree, starts box_factors.sh THERE, and pulls the results back.
#
# Usage: ROBOTRUTH_HOST=ubuntu@<ip> bash policyci/ops/run_policyci_box.sh <dir> [N] [SHARDS] [BIASES]
#   ROBOTRUTH_HOST=ubuntu@1.2.3.4 bash policyci/ops/run_policyci_box.sh pci_2 200 4 "0.02 0.05 0.10"
#
# Results land in D:\robotruth\results\<dir>\ ; the box is NOT terminated by this script.
set -euo pipefail
KEY="${ROBOTRUTH_SSH_KEY:-/c/Users/Lenovo/.ssh/id_ed25519_annotate_vast}"
HOST="${ROBOTRUTH_HOST:?set ROBOTRUTH_HOST=ubuntu@<box-ip>}"
REMOTE_DIR="${1:-pcf_1}"
N="${2:-200}"
SHARDS="${3:-4}"
BIASES="${4:-0.02 0.05 0.10}"
LOCAL_RESULTS="/d/robotruth/results/$REMOTE_DIR"
TGZ="$(mktemp -u /c/Users/Lenovo/AppData/Local/Temp/robotruth_sync_XXXXXX).tgz"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $HOST"

cd /d && tar --exclude=robotruth/.venv --exclude=robotruth/.git --exclude=robotruth/.env \
  --exclude='robotruth/examples' --exclude='robotruth/results' --exclude='__pycache__' \
  --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='robotruth/dist' -czf "$TGZ" robotruth
scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new "$TGZ" "$HOST:~/${REMOTE_DIR}.tgz"
rm -f "$TGZ"

$SSH "set -e; mkdir -p ~/$REMOTE_DIR && cd ~/$REMOTE_DIR && rm -rf robotruth && \
  tar -xzf ~/${REMOTE_DIR}.tgz -C . && rm -f ~/${REMOTE_DIR}.tgz && echo synced"

$SSH "bash ~/$REMOTE_DIR/robotruth/policyci/ops/box_factors.sh ~/$REMOTE_DIR '$N' '$SHARDS' '$BIASES'"

echo "=== pulling results to $LOCAL_RESULTS ==="
mkdir -p "$LOCAL_RESULTS"
scp -q -i "$KEY" "$HOST:~/$REMOTE_DIR/policyci_out.tgz" "$LOCAL_RESULTS/"
tar -xzf "$LOCAL_RESULTS/policyci_out.tgz" -C "$LOCAL_RESULTS"
rm -f "$LOCAL_RESULTS/policyci_out.tgz"
echo "pulled:"
find "$LOCAL_RESULTS" -maxdepth 2 -type d | head -20
find "$LOCAL_RESULTS" -name '*.mp4' | wc -l | xargs echo "videos:"

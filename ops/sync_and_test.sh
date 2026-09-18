#!/usr/bin/env bash
# Sync D:\robotruth to the Lambda box and run pytest THERE. Nothing runs on the local CPU.
# Usage: bash ops/sync_and_test.sh <remote_dir_name> [pytest args...]
#   bash ops/sync_and_test.sh rt_main                 # full suite
#   bash ops/sync_and_test.sh rt_judge tests/test_judge.py -q
set -euo pipefail
KEY="${ROBOTRUTH_SSH_KEY:-/c/Users/Lenovo/.ssh/id_ed25519_annotate_vast}"
HOST="${ROBOTRUTH_HOST:-ubuntu@129.80.241.17}"
REMOTE_DIR="${1:-rt_main}"; shift || true
PYTEST_ARGS="${*:--q}"
LOCAL_ROOT="/d/robotruth"
TGZ="$(mktemp -u /c/Users/Lenovo/AppData/Local/Temp/robotruth_sync_XXXXXX).tgz"
SSH="ssh -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 $HOST"

cd /d && tar --exclude=robotruth/.venv --exclude=robotruth/.git --exclude=robotruth/.env \
  --exclude='robotruth/examples/reports' --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='.ruff_cache' -czf "$TGZ" robotruth
scp -q -i "$KEY" -o StrictHostKeyChecking=accept-new "$TGZ" "$HOST:~/${REMOTE_DIR}.tgz"
rm -f "$TGZ"
$SSH "export PATH=\$HOME/.local/bin:\$PATH; set -e
  mkdir -p ~/$REMOTE_DIR && cd ~/$REMOTE_DIR
  rm -rf src tests ops examples pyproject.toml README.md STATUS.md 2>/dev/null || true
  tar -xzf ~/${REMOTE_DIR}.tgz --strip-components=1 && rm -f ~/${REMOTE_DIR}.tgz
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  [ -d .venv ] || uv venv .venv -q --python 3.12
  uv pip install -q -e '.[dev]' opencv-python-headless --python .venv/bin/python 2>&1 | tail -1 || true
  echo '--- pytest on' \$(hostname) '---'
  .venv/bin/python -m pytest $PYTEST_ARGS 2>&1 | tail -40"

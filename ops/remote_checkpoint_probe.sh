#!/usr/bin/env bash
# Generic checkpoint probe for a rented GPU box. Uses only public tools.
# Downloads public checkpoints, then writes for each repo:
#   probe/<repo_slug>/files.json      name, size, sha256 of every file
#   probe/<repo_slug>/configs/        every *.json under 2 MB, verbatim
#   probe/<repo_slug>/headers/        safetensors JSON headers (tensor names, shapes, dtypes) only
# Weights themselves never leave the box. Requires HF_TOKEN in the environment.
set -euo pipefail
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
OUT="$HOME/probe"
mkdir -p "$OUT"
python3 -m pip install -q --user huggingface_hub >/dev/null 2>&1 || true

REPOS=(
  "lerobot/smolvla_base"
  "lerobot/act_aloha_sim_transfer_cube_human"
  "lerobot/pi0"
  "lerobot/pi05_base"
  "nvidia/GR00T-N1.5-3B"
)

for repo in "${REPOS[@]}"; do
  slug="${repo//\//__}"
  echo "=== $repo"
  dir=$(python3 - "$repo" <<'PY'
import sys, os
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], token=os.environ.get("HF_TOKEN"),
      allow_patterns=["*.json","*.safetensors","*.yaml","*.yml","*.md","LICENSE",".eval_results/*","experiment_cfg/*"]))
PY
)
  echo "    at $dir"
  mkdir -p "$OUT/$slug/configs" "$OUT/$slug/headers"
  python3 - "$dir" "$OUT/$slug" <<'PY'
import sys, os, json, hashlib, struct
from pathlib import Path
src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
files = []
for p in sorted(src.rglob("*")):
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    rel = str(p.relative_to(src))
    files.append({"name": rel, "size": p.stat().st_size, "sha256": h.hexdigest()})
    if p.suffix == ".json" and p.stat().st_size < 2_000_000:
        dst = out / "configs" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
    if p.suffix == ".safetensors":
        with p.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        dst = out / "headers" / (rel.replace("/", "__") + ".header.json")
        dst.write_text(json.dumps(hdr, indent=1))
(out / "files.json").write_text(json.dumps({"source": str(src), "files": files}, indent=1))
print(f"    {len(files)} files hashed")
PY
done
echo "=== nvidia-smi"; nvidia-smi --query-gpu=name,memory.total --format=csv || true
echo "=== done"; du -sh "$OUT"

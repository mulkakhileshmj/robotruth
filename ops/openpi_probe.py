"""Download a real openpi checkpoint from the public GCS bucket and validate the extractor.

openpi (Physical Intelligence) publishes base checkpoints in gs://openpi-assets. The layout is
  checkpoints/<name>/params/...            orbax weights
  checkpoints/<name>/assets/<id>/norm_stats.json
This script lists the bucket over the JSON API, downloads one checkpoint (anonymous HTTP,
no gsutil), runs `extract_spec` on it, and writes the manifest plus a short report.

Usage (on the GPU box): python ops/openpi_probe.py [checkpoint_name] [out_dir]
Defaults: pi0_base, ~/validation/openpi
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BUCKET = "openpi-assets"


def list_objects(prefix: str) -> list[dict]:
    items, token = [], None
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
               f"prefix={urllib.parse.quote(prefix)}&maxResults=1000")
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        items += data.get("items", [])
        token = data.get("nextPageToken")
        if not token:
            return items


def download(items: list[dict], dest: Path) -> int:
    total = 0
    for i, it in enumerate(items):
        rel = it["name"]
        out = dest / rel
        size = int(it.get("size", 0))
        if out.exists() and out.stat().st_size == size:
            total += size
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(rel)}"
        t = time.time()
        with urllib.request.urlopen(url, timeout=600) as r, out.open("wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        total += size
        print(f"  [{i+1}/{len(items)}] {rel.split('/')[-1][:60]:60s} {size/1e6:9.1f} MB {time.time()-t:5.1f}s", flush=True)
    return total


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "pi0_base"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path.home() / "validation" / "openpi"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"checkpoints/{name}/"
    items = list_objects(prefix)
    if not items:
        print(f"no objects under gs://{BUCKET}/{prefix}")
        return 1
    total = sum(int(i.get("size", 0)) for i in items)
    print(f"{len(items)} objects, {total/1e9:.2f} GB under {prefix}")
    dest = Path.home() / "openpi_ckpts"
    downloaded = download(items, dest)
    print(f"downloaded or present: {downloaded/1e9:.2f} GB")
    ckpt_dir = dest / "checkpoints" / name

    from robotruth.contract import check_contract, extract_spec
    spec = extract_spec(ckpt_dir, role="evaluated")
    manifest = out_dir / f"{name}.execspec.json"
    spec.to_json(manifest)
    self_check = check_contract(spec, spec)
    report = {
        "checkpoint": name, "family": spec.policy.family,
        "detected_family_correct": spec.policy.family == "openpi",
        "n_files": len(items), "bytes": total,
        "weights_sha256": spec.policy.weights_sha256, "weights_bytes": spec.policy.weights_bytes,
        "weights_files_counted": len(spec.policy.weights_files),
        "norm_type": spec.action.normalization.type,
        "stats_sha256": spec.action.normalization.stats_sha256,
        "stats_source": spec.action.normalization.stats_source,
        "action_dim": spec.action.dim,
        "unverified": spec.unverified, "notes": spec.notes,
        "self_check": self_check.summary(),
    }
    (out_dir / f"{name}.report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Resolve and fetch a broad benchmark set of public LeRobot-format datasets onto the box.

Metadata and parquet only, never video. Repo names for the big families are resolved at
runtime through the Hub API so renames do not break the script, and each repo's parquet
payload is size-checked before download. Failures are printed and skipped.

Usage: python ops/benchmark_fetch.py <dest_root> [--cap-gb 6] [--per-family 10]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

NO_VIDEO = ["meta/**", "data/**", "README.md", "*.json", "*.jsonl", "*.csv", "*.yaml"]

# Proven labelled or intervention-rich sets from the 2026-09-18 sweep.
EXPLICIT = [
    ("kantine/BotFails", ["**/meta/**", "**/data/**", "**/labels/**", "*.md", "*.csv", "*.json"]),
    ("lerobot/example_hil_serl_dataset", NO_VIDEO),
    ("k-chan-l/rollout_dice_color_sort_dagger", NO_VIDEO),
    ("TheMuz/rollout_kirby_dagger_v2_iter1_20260507_143451", NO_VIDEO),
    ("recast-robotics/rollout_dagger_r1_s2_b0_20260915_121327", NO_VIDEO),
    ("Elvinky/so101-fold-clothes-dagger-20260909", NO_VIDEO),
    ("Elvinky/pi05-piperx-7h-demo-dagger-full-episodes-20260914", NO_VIDEO),
    ("jpizarrom/hilserl_so100_grocery_so100_2025112223_20", NO_VIDEO),
    ("lerobot/svla_so100_pickplace", NO_VIDEO),
    ("lerobot/svla_so101_pickplace", NO_VIDEO),
    ("sixpigs1/so100_pick_cube_in_box", NO_VIDEO),
    ("jccj/so100_block_in_cup", NO_VIDEO),
    ("lerobot/aloha_static_screw_driver", NO_VIDEO),
    ("lerobot/droid_1.0.1", ["meta/**", "data/chunk-000/file-000.parquet", "README.md"]),
]

# Families resolved at runtime: (author, name substring filter, note)
FAMILIES = [
    ("IPEC-COMMUNITY", "lerobot", "OpenX conversions to LeRobot format"),
    ("BAAI-DataCube", "", "RoboMIND and AgiBotWorld task ports (LeRobot v3)"),
]


def parquet_gb(api: HfApi, repo: str) -> float:
    """Parquet payload size from the datasets-server size endpoint (one fast request)."""
    import requests

    try:
        r = requests.get("https://datasets-server.huggingface.co/size",
                         params={"dataset": repo}, timeout=20)
        if r.status_code != 200:
            return -1.0
        d = r.json().get("size", {}).get("dataset", {})
        b = d.get("num_bytes_parquet_files") or d.get("num_bytes_original_files") or 0
        return b / 1e9 if b else -1.0
    except Exception:
        return -1.0


def fetch(repo: str, pats, dest: Path) -> None:
    t = time.time()
    try:
        p = snapshot_download(repo, repo_type="dataset", allow_patterns=pats,
                              local_dir=str(dest / repo.replace("/", "__")))
        size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9
        print(f"OK   {repo:70s} {size:7.2f} GB {time.time()-t:6.0f}s", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {repo:70s} {type(e).__name__}: {str(e)[:120]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--cap-gb", type=float, default=6.0)
    ap.add_argument("--per-family", type=int, default=10)
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    for repo, pats in EXPLICIT:
        fetch(repo, pats, dest)

    for author, needle, note in FAMILIES:
        print(f"-- resolving {author} ({note})", flush=True)
        try:
            cands = [d.id for d in api.list_datasets(author=author, limit=300)
                     if needle in d.id.lower()]
        except Exception as e:  # noqa: BLE001
            print(f"FAIL list {author}: {e}", flush=True)
            continue
        sized = []
        for repo in cands:
            gb = parquet_gb(api, repo)
            print(f"   size {repo}: {gb:.2f} GB", flush=True)
            if 0 < gb <= args.cap_gb:
                sized.append((gb, repo))
            if len(sized) >= 3 * args.per_family:
                break
        sized.sort()
        for gb, repo in sized[: args.per_family]:
            print(f"   picked {repo} ({gb:.2f} GB parquet)", flush=True)
            fetch(repo, NO_VIDEO, dest)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch public real-robot datasets onto the GPU box, metadata and parquet only (no video).

Usage: python ops/fetch_real_datasets.py <dest_root>
"""
import os, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

DEST = Path(sys.argv[1]); DEST.mkdir(parents=True, exist_ok=True)
NO_VIDEO = ["meta/**", "data/**", "README.md", "*.json", "*.jsonl", "*.csv", "*.yaml"]
JOBS = [
    ("lerobot/example_hil_serl_dataset", NO_VIDEO),
    ("k-chan-l/rollout_dice_color_sort_dagger", NO_VIDEO),
    ("TheMuz/rollout_kirby_dagger_v2_iter1_20260507_143451", NO_VIDEO),
    ("recast-robotics/rollout_dagger_r1_s2_b0_20260915_121327", NO_VIDEO),
    ("Elvinky/so101-fold-clothes-dagger-20260909", NO_VIDEO),
    ("Elvinky/pi05-piperx-7h-demo-dagger-full-episodes-20260914", NO_VIDEO),
    ("jpizarrom/hilserl_so100_grocery_so100_2025112223_20", NO_VIDEO),
    ("kantine/BotFails", ["**/meta/**", "**/data/**", "**/labels/**", "*.md", "*.csv", "*.json"]),
    ("lerobot/svla_so100_pickplace", NO_VIDEO),
    ("lerobot/svla_so101_pickplace", NO_VIDEO),
    ("sixpigs1/so100_pick_cube_in_box", NO_VIDEO),
    ("jccj/so100_block_in_cup", NO_VIDEO),
    ("lerobot/aloha_static_screw_driver", NO_VIDEO),
    ("lerobot/droid_1.0.1", ["meta/**", "data/chunk-000/file-000.parquet", "README.md"]),
    ("cons909/V1-LeRobot-SO101-Eval-Videos", ["*.csv", "*.json", "*.md", "**/*.csv"]),
    ("RoboArena/DataDump_02-03-2026", ["global_metadata.yaml", "README.md", "**/metadata.yaml"]),
]
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
for repo, pats in JOBS:
    t = time.time()
    try:
        p = snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=str(DEST / repo.replace("/", "__")))
        size = sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9
        print(f"OK   {repo:60s} {size:7.2f} GB {time.time()-t:6.0f}s", flush=True)
    except Exception as e:
        print(f"FAIL {repo:60s} {type(e).__name__}: {str(e)[:120]}", flush=True)
print("done")

"""Expose nested LeRobot datasets (like BotFails subsets) as top-level dataset dirs.

validate_real.py only looks one level deep for meta/info.json. Some repos ship many
datasets inside one repo; this symlinks each nested dataset into the root so every one
is validated on its own.

Usage: python ops/flatten_nested_datasets.py <root>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
made = 0
for top in sorted(p for p in root.iterdir() if p.is_dir()):
    if (top / "meta" / "info.json").exists():
        continue
    for info in sorted(top.rglob("meta/info.json")):
        ds = info.parent.parent
        link = root / f"{top.name}__{ds.relative_to(top).as_posix().replace('/', '_')}"
        if not link.exists():
            link.symlink_to(ds.resolve())
            made += 1
print(f"linked {made} nested datasets")

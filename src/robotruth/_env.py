"""Tiny .env loader that tolerates PowerShell's UTF-16 output. Never prints values."""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path | str = ".env") -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            text = raw.decode(enc)
            if "=" in text:
                break
        except UnicodeDecodeError:
            continue
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip().lstrip("﻿")
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in out.items():
        os.environ.setdefault(k, v)
    return out


def hf_token() -> str | None:
    env = load_env(Path(__file__).resolve().parents[2] / ".env")
    return env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")

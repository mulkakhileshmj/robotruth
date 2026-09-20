"""Deterministic scenario identity.

A scenario is a value, not a run: a parameter vector plus a schema version, content-hashed.
"37 new failures" only means something if scenario N is the identical scene setup for both
policies, so scenarios and batteries are immutable and content-addressed. Comparisons across
mismatched simulator pins are refused elsewhere (see regression.py), fail-closed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

SCENARIO_SCHEMA_VERSION = "0.1"


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def content_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj)).hexdigest()


@dataclass(frozen=True)
class Scenario:
    """One fully-specified scene. `params` must contain everything the backend needs to
    reconstruct the identical initial state (for seed-driven backends, the reset seed)."""

    index: int
    params: dict
    schema_version: str = SCENARIO_SCHEMA_VERSION

    @property
    def hash(self) -> str:
        return content_hash({"params": self.params, "schema_version": self.schema_version})

    @property
    def short(self) -> str:
        return self.hash[:12]

    def to_dict(self) -> dict:
        return {"index": self.index, "params": self.params,
                "schema_version": self.schema_version, "hash": self.hash}


@dataclass(frozen=True)
class Battery:
    """An immutable, content-addressed list of scenarios: the standing test suite.

    `battery_hash` covers the task, backend id, schema version and every scenario hash,
    so "ran battery <hash>" is a reproducible claim.
    """

    name: str
    task: str
    backend_id: str
    scenarios: tuple[Scenario, ...]
    schema_version: str = SCENARIO_SCHEMA_VERSION

    @property
    def battery_hash(self) -> str:
        return content_hash({
            "task": self.task,
            "backend_id": self.backend_id,
            "schema_version": self.schema_version,
            "scenarios": [s.hash for s in self.scenarios],
        })

    def __len__(self) -> int:
        return len(self.scenarios)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        head = {"kind": "policyci.battery", "name": self.name, "task": self.task,
                "backend_id": self.backend_id, "schema_version": self.schema_version,
                "n": len(self.scenarios), "battery_hash": self.battery_hash}
        with path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(head) + "\n")
            for s in self.scenarios:
                f.write(json.dumps(s.to_dict()) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Battery":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        head = json.loads(lines[0])
        if head.get("kind") != "policyci.battery":
            raise ValueError(f"{path} is not a policyci battery file")
        scenarios = []
        for line in lines[1:]:
            if not line.strip():
                continue
            d = json.loads(line)
            s = Scenario(index=d["index"], params=d["params"], schema_version=d["schema_version"])
            if s.hash != d["hash"]:
                raise ValueError(f"scenario {d['index']} hash mismatch: file edited or corrupt")
            scenarios.append(s)
        bat = cls(name=head["name"], task=head["task"], backend_id=head["backend_id"],
                  scenarios=tuple(scenarios), schema_version=head["schema_version"])
        if bat.battery_hash != head["battery_hash"]:
            raise ValueError("battery hash mismatch: file edited or corrupt")
        return bat


def sample_seed_battery(name: str, task: str, backend_id: str, n: int, base_seed: int = 0) -> Battery:
    """v1 battery for seed-driven backends: each scenario is one deterministic reset seed.

    Seeds are drawn from a PCG64 stream keyed by base_seed, so the battery itself is a pure
    function of (name inputs, n, base_seed) and re-sampling reproduces it bit for bit.
    """
    rng = np.random.Generator(np.random.PCG64(base_seed))
    seeds = rng.integers(0, 2**31 - 1, size=n)
    scenarios = tuple(
        Scenario(index=i, params={"reset_seed": int(s)}) for i, s in enumerate(seeds)
    )
    return Battery(name=name, task=task, backend_id=backend_id, scenarios=scenarios)

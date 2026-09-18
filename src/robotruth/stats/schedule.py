"""Blinded, interleaved rollout schedules.

Running all of policy A in the morning and all of policy B after lunch confounds the
comparison with lighting, operator fatigue and actuator temperature. Interleave, block by
condition, and hide the policy identity from the operator until the outcome is recorded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Slot:
    index: int
    block: int
    condition: str
    policy: str
    blind_label: str
    pair_id: str


@dataclass
class Schedule:
    slots: list[Slot] = field(default_factory=list)
    key: dict[str, str] = field(default_factory=dict)  # blind_label -> policy

    def to_rows(self) -> list[dict]:
        return [dict(index=s.index, block=s.block, condition=s.condition, blind_label=s.blind_label, pair_id=s.pair_id) for s in self.slots]

    def unblind(self) -> list[dict]:
        return [dict(index=s.index, block=s.block, condition=s.condition, policy=s.policy, pair_id=s.pair_id) for s in self.slots]


def interleaved_schedule(policies: list[str], conditions: list[str], repeats: int = 1, seed: int = 0) -> Schedule:
    """Each block is one condition; every policy appears once per block in random order.

    Slots sharing a `pair_id` are matched observations for paired tests.
    """
    rng = np.random.default_rng(seed)
    labels = [chr(ord("A") + i) for i in range(len(policies))]
    perm = rng.permutation(len(policies))
    key = {labels[i]: policies[perm[i]] for i in range(len(policies))}
    inv = {v: k for k, v in key.items()}
    sched = Schedule(key=key)
    i = 0
    block = 0
    for _ in range(repeats):
        for cond in rng.permutation(conditions):
            order = rng.permutation(policies)
            pair_id = hashlib.sha1(f"{seed}-{block}-{cond}".encode()).hexdigest()[:8]
            for pol in order:
                sched.slots.append(Slot(i, block, str(cond), str(pol), inv[str(pol)], pair_id))
                i += 1
            block += 1
    return sched

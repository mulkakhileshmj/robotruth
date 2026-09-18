"""Attaching the Guard to a policy server, and replaying recorded episodes.

Nothing here imports lerobot or openpi. The Guard only needs two numpy arrays per
inference, so the attachment is a few lines inside whichever server loop you run. The
recipes below are documentation, not code that runs against those packages.

LeRobot async policy server (lerobot.async_inference.policy_server, LeRobot 0.6):
    The server calls the policy once per observation and returns an action chunk. Wrap the
    call that produces the chunk:

        hook = attach(guard)                       # PolicyServerHook
        ...
        chunk = policy.predict_action_chunk(batch)  # [1, chunk, D] tensor
        feats = policy_features(policy)             # see below
        event = hook(feats, chunk[0].cpu().numpy(), t=time.monotonic())
        if event.level == "stop": send zero-velocity / hold command and raise the flag
        elif event.level in ("slow", "handover"): scale the chunk or request teleop

    Feature vector: register a forward hook on the module that feeds the action head (for
    ACT the transformer decoder output, for SmolVLA or pi0 the last hidden state of the
    action expert) and keep its mean over tokens as a [F] vector. Any fixed-size vector
    that is a deterministic function of the observation works; the Guard is agnostic.

openpi websocket server (openpi.serving.websocket_policy_server):
    Wrap the policy passed to the server so that infer(obs) also runs the hook:

        class GuardedPolicy:
            def __init__(self, policy, hook): self.policy, self.hook = policy, hook
            def infer(self, obs):
                out = self.policy.infer(obs)                       # dict with "actions" [chunk, D]
                feats = out.get("features")                        # add to the model output if absent
                event = self.hook(feats, np.asarray(out["actions"]), t=time.monotonic())
                out["guard"] = event.to_dict()
                return out

    Call guard.start_episode() on a reset message and guard.end_episode(truth) when the
    operator labels the episode. That is what makes the false alarm accounting possible.

OfflineReplay runs a Guard over a recorded episode saved as npz with keys
    features   [T, F]        (optional if the guard has no feature scorer)
    actions    [T, chunk, D] (optional if the guard has no action scorer)
    timestamps [T]           seconds (optional; step index times guard.dt otherwise)
    state      [T, D]        measured joints (optional, for hard limits)
    truth      scalar        "success" | "failure" | 0 | 1 (optional)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from robotruth.guard.metrics import GuardEpisode
from robotruth.guard.monitor import Guard, GuardEvent


class PolicyServerHook(Protocol):
    """Callable a policy server invokes once per inference.

    features: the policy's feature vector for this inference, [F], or None.
    action_chunk: the predicted actions, [chunk, D], or None.
    t: wall-clock or episode time in seconds; None uses step index times guard.dt.
    state: measured joint positions [D] for hard-limit checks, or None.
    Returns the GuardEvent for this step; act on event.level.
    """

    def __call__(self, features: Optional[np.ndarray], action_chunk: Optional[np.ndarray],
                 t: Optional[float] = None, state: Optional[np.ndarray] = None) -> GuardEvent: ...


def attach(guard: Guard) -> PolicyServerHook:
    """Return a PolicyServerHook bound to `guard`. Tensors are converted with np.asarray."""

    def hook(features, action_chunk, t=None, state=None) -> GuardEvent:
        f = None if features is None else np.asarray(_to_numpy(features), dtype=np.float64).ravel()
        a = None if action_chunk is None else np.asarray(_to_numpy(action_chunk), dtype=np.float64)
        if a is not None and a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        s = None if state is None else np.asarray(_to_numpy(state), dtype=np.float64).ravel()
        return guard.step(features=f, action_chunk=a, t=t, state=s)

    return hook


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return x


def load_episode_npz(path: Path | str) -> dict[str, Any]:
    """Read an episode npz into a plain dict of numpy arrays plus a parsed truth label."""
    with np.load(Path(path), allow_pickle=False) as z:
        ep: dict[str, Any] = {k: z[k] for k in z.files}
    if "truth" in ep:
        raw = ep["truth"]
        ep["truth"] = raw.item() if getattr(raw, "shape", ()) == () else raw.tolist()
        if isinstance(ep["truth"], bytes):
            ep["truth"] = ep["truth"].decode()
    return ep


def save_episode_npz(path: Path | str, features: Optional[np.ndarray] = None, actions: Optional[np.ndarray] = None,
                     timestamps: Optional[np.ndarray] = None, state: Optional[np.ndarray] = None,
                     truth: Optional[Any] = None) -> Path:
    """Write an episode npz in the layout OfflineReplay reads."""
    arrays: dict[str, Any] = {}
    if features is not None:
        arrays["features"] = np.asarray(features, dtype=np.float64)
    if actions is not None:
        arrays["actions"] = np.asarray(actions, dtype=np.float64)
    if timestamps is not None:
        arrays["timestamps"] = np.asarray(timestamps, dtype=np.float64)
    if state is not None:
        arrays["state"] = np.asarray(state, dtype=np.float64)
    if truth is not None:
        arrays["truth"] = np.asarray(str(truth))
    if not arrays:
        raise ValueError("nothing to save")
    p = Path(path)
    np.savez(p, **arrays)
    return p


@dataclass
class ReplayResult:
    episode: GuardEpisode
    events: list[GuardEvent] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        e = self.episode
        truth = {True: "success", False: "failure", None: "unlabelled"}[e.truth]
        if e.first_alert_t is None:
            alert = "no alert"
        else:
            alert = f"first alert at {e.first_alert_t:.2f}s of {e.duration_s:.2f}s (lead {e.lead_time_s:.2f}s)"
        return f"{e.episode_id}: max level {e.max_level}, {alert}, {e.n_alert_onsets} onsets, truth={truth}"


@dataclass
class OfflineReplay:
    """Run a Guard over a recorded episode as if it were live."""

    guard: Guard

    def run(self, episode: dict[str, Any] | Path | str, truth: Optional[Any] = None,
            episode_id: Optional[str] = None) -> ReplayResult:
        ep = load_episode_npz(episode) if isinstance(episode, (str, Path)) else dict(episode)
        if episode_id is None and isinstance(episode, (str, Path)):
            episode_id = Path(episode).stem
        feats = ep.get("features")
        acts = ep.get("actions")
        ts = ep.get("timestamps")
        st = ep.get("state")
        if feats is None and acts is None:
            raise ValueError("episode needs 'features' or 'actions'")
        n = feats.shape[0] if feats is not None else acts.shape[0]
        for name, arr in (("features", feats), ("actions", acts), ("timestamps", ts), ("state", st)):
            if arr is not None and arr.shape[0] != n:
                raise ValueError(f"'{name}' has {arr.shape[0]} steps but the episode has {n}")
        if truth is None:
            truth = ep.get("truth")
        self.guard.start_episode(episode_id)
        events = []
        for i in range(n):
            events.append(self.guard.step(
                features=None if feats is None else feats[i],
                action_chunk=None if acts is None else acts[i],
                t=None if ts is None else float(ts[i]),
                state=None if st is None else st[i]))
        rec = self.guard.end_episode(truth, episode_id=episode_id)
        return ReplayResult(rec, events)

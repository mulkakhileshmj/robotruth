"""Runtime failure scores for learned robot policies.

Each scorer turns one policy inference into a single number that is small when the policy
is behaving like it did on successful rollouts and large when it is not. None of them needs
failure data: they are fitted on nominal (successful) rollouts only, which is the whole
point of the FAIL-Detect recipe (RSS 2025, arXiv 2503.08558).

Scorers here:

- MahalanobisScorer: squared Mahalanobis distance of the policy's last-layer feature vector
  from the nominal feature cloud, with Ledoit-Wolf shrinkage of the covariance so it works
  when features outnumber rollouts. This is the VLA-FAIL feature score (arXiv 2606.21386).
- ChunkConsistencyScorer: how much the newly predicted action chunk disagrees with the
  previous chunk over the steps they both cover. A policy that keeps changing its mind is
  a policy in trouble. VLA-FAIL action score.
- ActionStatsScorer: magnitude and spread of the current chunk relative to nominal. Cheap,
  catches freezes (spread collapses) and lunges (magnitude blows up). ActProbe style.
- CompositeScorer: weighted sum of standardised component scores.

Every scorer follows the same small protocol: fit(nominal), score(x), reset() for per-episode
state, to_dict() and from_dict() for saving into the guard json, and score_episode(episode)
which runs the scorer over a whole recorded episode and returns one score per step.

numpy only in the core path. If torch is importable and a CUDA device is present, the
Mahalanobis scorer will score batches on the GPU; torch is never required.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

_EPS = 1e-12


def _torch_cuda():
    """Return the torch module if it is importable and a CUDA device exists, else None."""
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return torch
    except Exception:
        return None
    return None


def ledoit_wolf_covariance(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance of the rows of x, shape [N, F].

    Shrinks the sample covariance S toward mu * I where mu is the average variance, with
    the intensity from Ledoit and Wolf (2004), "A well-conditioned estimator for
    large-dimensional covariance matrices". Returns (covariance, shrinkage_intensity).
    Implemented directly in numpy so scikit-learn is not needed.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected a 2D array [N, F]")
    n, p = x.shape
    if n < 2:
        raise ValueError("need at least 2 nominal samples")
    xc = x - x.mean(axis=0, keepdims=True)
    s = xc.T @ xc / n
    mu = float(np.trace(s)) / p
    eye = np.eye(p)
    delta2 = float(np.sum((s - mu * eye) ** 2)) / p
    sq_norms = np.sum(xc * xc, axis=1)
    beta2_bar = (float(np.sum(sq_norms ** 2)) / (n * n) - float(np.sum(s * s)) / n) / p
    beta2 = min(max(beta2_bar, 0.0), delta2)
    shrink = 0.0 if delta2 <= _EPS else beta2 / delta2
    cov = shrink * mu * eye + (1.0 - shrink) * s
    return cov, float(shrink)


class Scorer:
    """Base class. Subclasses set `needs` to "features" or "action_chunk"."""

    kind: str = "base"
    needs: str = "features"

    def __init__(self) -> None:
        self.fitted = False

    def fit(self, nominal: Any) -> "Scorer":
        raise NotImplementedError

    def score(self, x: Any) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear per-episode state. Called by the Guard at start_episode()."""

    def score_step(self, features: Optional[np.ndarray] = None,
                   action_chunk: Optional[np.ndarray] = None) -> tuple[Optional[float], dict[str, float]]:
        """Score one policy inference from whichever input this scorer needs.

        Returns (score, components). score is None when the required input is missing.
        """
        x = features if self.needs == "features" else action_chunk
        if x is None:
            return None, {}
        s = float(self.score(np.asarray(x, dtype=np.float64)))
        return s, {self.kind: s}

    def score_episode(self, episode: dict[str, Any]) -> np.ndarray:
        """One score per step over a recorded episode dict with 'features' and/or 'actions'."""
        self.reset()
        key = "features" if self.needs == "features" else "actions"
        arr = episode.get(key)
        if arr is None:
            raise ValueError(f"episode has no '{key}' array, which {self.kind} needs")
        arr = np.asarray(arr, dtype=np.float64)
        out = np.empty(arr.shape[0], dtype=np.float64)
        for i in range(arr.shape[0]):
            s, _ = self.score_step(**{self.needs: arr[i]})
            out[i] = s if s is not None else 0.0
        return out

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Scorer":
        raise NotImplementedError


class MahalanobisScorer(Scorer):
    """Squared Mahalanobis distance of a feature vector from the nominal feature cloud.

    fit(features) takes [N, F] feature vectors collected from successful rollouts (for
    example the input to the policy's action head). score(x) accepts one vector [F] and
    returns a float, or a batch [B, F] and returns [B]. Under a Gaussian nominal model the
    score is roughly chi-squared with F degrees of freedom, so it is not a probability;
    the conformal calibrator turns it into one.

    device: "auto" uses torch on CUDA for batches when available, "cpu" forces numpy.
    last_backend records which path scored the most recent call ("numpy" or "torch_cuda").
    """

    kind = "mahalanobis"
    needs = "features"

    def __init__(self, device: str = "auto", min_batch_for_torch: int = 2) -> None:
        super().__init__()
        self.device = device
        self.min_batch_for_torch = int(min_batch_for_torch)
        self.mean: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None
        self.shrinkage: Optional[float] = None
        self.n_fit: int = 0
        self.last_backend: str = "numpy"
        self._torch_precision = None
        self._torch_mean = None

    def fit(self, nominal: np.ndarray) -> "MahalanobisScorer":
        x = np.asarray(nominal, dtype=np.float64)
        cov, shrink = ledoit_wolf_covariance(x)
        self.mean = x.mean(axis=0)
        self.precision = np.linalg.pinv(cov, hermitian=True)
        self.shrinkage = shrink
        self.n_fit = int(x.shape[0])
        self._torch_precision = None
        self._torch_mean = None
        self.fitted = True
        return self

    def _score_numpy(self, x: np.ndarray) -> np.ndarray:
        d = x - self.mean
        return np.einsum("bi,ij,bj->b", d, self.precision, d)

    def _score_torch(self, torch, x: np.ndarray) -> np.ndarray:
        if self._torch_precision is None:
            self._torch_precision = torch.as_tensor(self.precision, dtype=torch.float64, device="cuda")
            self._torch_mean = torch.as_tensor(self.mean, dtype=torch.float64, device="cuda")
        xt = torch.as_tensor(x, dtype=torch.float64, device="cuda")
        d = xt - self._torch_mean
        out = torch.einsum("bi,ij,bj->b", d, self._torch_precision, d)
        return out.detach().cpu().numpy()

    def score(self, x: np.ndarray):
        if not self.fitted:
            raise RuntimeError("MahalanobisScorer.fit() first")
        x = np.asarray(x, dtype=np.float64)
        single = x.ndim == 1
        xb = x[None, :] if single else x
        if xb.shape[1] != self.mean.shape[0]:
            raise ValueError(f"feature size {xb.shape[1]} does not match fitted size {self.mean.shape[0]}")
        torch = _torch_cuda() if (self.device == "auto" or self.device == "cuda") else None
        if torch is not None and xb.shape[0] >= self.min_batch_for_torch:
            out = self._score_torch(torch, xb)
            self.last_backend = "torch_cuda"
        else:
            out = self._score_numpy(xb)
            self.last_backend = "numpy"
        out = np.maximum(out, 0.0)
        return float(out[0]) if single else out

    def score_episode(self, episode: dict[str, Any]) -> np.ndarray:
        feats = episode.get("features")
        if feats is None:
            raise ValueError("episode has no 'features' array, which mahalanobis needs")
        return np.asarray(self.score(np.asarray(feats, dtype=np.float64)), dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "device": self.device, "min_batch_for_torch": self.min_batch_for_torch,
                "mean": None if self.mean is None else self.mean.tolist(),
                "precision": None if self.precision is None else self.precision.tolist(),
                "shrinkage": self.shrinkage, "n_fit": self.n_fit}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MahalanobisScorer":
        s = cls(device=d.get("device", "auto"), min_batch_for_torch=d.get("min_batch_for_torch", 2))
        if d.get("mean") is not None:
            s.mean = np.asarray(d["mean"], dtype=np.float64)
            s.precision = np.asarray(d["precision"], dtype=np.float64)
            s.shrinkage = d.get("shrinkage")
            s.n_fit = int(d.get("n_fit", 0))
            s.fitted = True
        return s


class ChunkConsistencyScorer(Scorer):
    """Disagreement between consecutive action chunks over the steps they both cover.

    The policy emits a chunk of `chunk` future actions [chunk, D] every `stride` control
    steps. Chunk t and chunk t-1 both describe steps stride .. chunk-1 of the older chunk,
    so prev[stride:] and cur[:chunk - stride] should agree. The raw score is the mean L2
    distance over that overlap. When stride >= chunk there is no overlap and the raw score
    falls back to the jump between the last action of the previous chunk and the first
    action of the new one.

    After fit() the score is standardised by the nominal mean and standard deviation of the
    raw score, so 0 means "as consistent as a typical successful rollout" and 3 means three
    nominal standard deviations worse. The first chunk of an episode scores 0 (no evidence).
    """

    kind = "chunk_consistency"
    needs = "action_chunk"

    def __init__(self, stride: Optional[int] = None) -> None:
        super().__init__()
        self.stride = stride
        self.raw_mean: Optional[float] = None
        self.raw_std: Optional[float] = None
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def raw_score(self, prev: np.ndarray, cur: np.ndarray) -> float:
        prev = np.asarray(prev, dtype=np.float64)
        cur = np.asarray(cur, dtype=np.float64)
        if prev.shape != cur.shape or prev.ndim != 2:
            raise ValueError("chunks must share shape [chunk, D]")
        chunk = prev.shape[0]
        stride = self.stride if self.stride is not None else max(1, chunk // 2)
        if stride < chunk:
            a, b = prev[stride:], cur[: chunk - stride]
        else:
            a, b = prev[-1:], cur[:1]
        return float(np.mean(np.linalg.norm(a - b, axis=1)))

    def raw_sequence(self, chunks: np.ndarray) -> np.ndarray:
        chunks = np.asarray(chunks, dtype=np.float64)
        if chunks.ndim != 3:
            raise ValueError("expected chunks [T, chunk, D]")
        out = np.zeros(chunks.shape[0], dtype=np.float64)
        for i in range(1, chunks.shape[0]):
            out[i] = self.raw_score(chunks[i - 1], chunks[i])
        return out

    def fit(self, nominal) -> "ChunkConsistencyScorer":
        """nominal: one array [T, chunk, D] or a list of them (one per episode)."""
        seqs = [nominal] if isinstance(nominal, np.ndarray) and nominal.ndim == 3 else list(nominal)
        raws = np.concatenate([self.raw_sequence(np.asarray(s))[1:] for s in seqs if len(s) > 1])
        if raws.size < 2:
            raise ValueError("need at least two consecutive chunks to fit")
        self.raw_mean = float(raws.mean())
        self.raw_std = float(max(raws.std(ddof=1), _EPS))
        self.fitted = True
        self.reset()
        return self

    def score(self, x: np.ndarray) -> float:
        if not self.fitted:
            raise RuntimeError("ChunkConsistencyScorer.fit() first")
        cur = np.asarray(x, dtype=np.float64)
        if self._prev is None:
            self._prev = cur
            return 0.0
        raw = self.raw_score(self._prev, cur)
        self._prev = cur
        return (raw - self.raw_mean) / self.raw_std

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "stride": self.stride, "raw_mean": self.raw_mean, "raw_std": self.raw_std}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ChunkConsistencyScorer":
        s = cls(stride=d.get("stride"))
        if d.get("raw_mean") is not None:
            s.raw_mean, s.raw_std, s.fitted = float(d["raw_mean"]), float(d["raw_std"]), True
        return s


class ActionStatsScorer(Scorer):
    """Magnitude and spread of the current action chunk relative to nominal.

    Two raw statistics per chunk [chunk, D]: the mean L2 norm of the actions (magnitude)
    and the mean over dimensions of the variance across the chunk (spread). After fit the
    score is the average of the absolute z-scores of the two, so a chunk that is far too
    big, far too small, frozen or jittery all score high.
    """

    kind = "action_stats"
    needs = "action_chunk"

    def __init__(self) -> None:
        super().__init__()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    @staticmethod
    def raw_stats(chunk: np.ndarray) -> np.ndarray:
        c = np.asarray(chunk, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("expected a chunk [chunk, D]")
        mag = float(np.mean(np.linalg.norm(c, axis=1)))
        spread = float(np.mean(np.var(c, axis=0))) if c.shape[0] > 1 else 0.0
        return np.array([mag, spread])

    def fit(self, nominal: np.ndarray) -> "ActionStatsScorer":
        """nominal: chunks [N, chunk, D] pooled over nominal episodes."""
        chunks = np.asarray(nominal, dtype=np.float64)
        if chunks.ndim != 3 or chunks.shape[0] < 2:
            raise ValueError("expected at least two chunks [N, chunk, D]")
        stats = np.stack([self.raw_stats(c) for c in chunks])
        self.mean = stats.mean(axis=0)
        self.std = np.maximum(stats.std(axis=0, ddof=1), _EPS)
        self.fitted = True
        return self

    def score(self, x: np.ndarray) -> float:
        if not self.fitted:
            raise RuntimeError("ActionStatsScorer.fit() first")
        z = (self.raw_stats(x) - self.mean) / self.std
        return float(np.mean(np.abs(z)))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "mean": None if self.mean is None else self.mean.tolist(),
                "std": None if self.std is None else self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ActionStatsScorer":
        s = cls()
        if d.get("mean") is not None:
            s.mean, s.std, s.fitted = np.asarray(d["mean"]), np.asarray(d["std"]), True
        return s


class StagnationScorer(Scorer):
    """Near-zero motion of the action stream over a sliding window (a stalled policy).

    A frozen or stalled policy keeps emitting the same chunk. That stream is perfectly
    self-consistent and every single chunk is in-distribution, so the consistency, action
    statistics and Mahalanobis scorers never fire on it. This scorer watches motion instead:
    the raw motion at step t is the L2 distance between the mean of chunk t and the mean of
    chunk t-1, averaged over the last `window` steps. Averaging over the window (rather than
    summing) keeps early steps with a partial window on the same scale.

    After fit() the score is how many nominal standard deviations the log motion sits below
    the nominal mean log motion, clipped at 0, so a robot moving as much as usual (or more)
    scores 0 and a robot that has stopped scores high. Logs are used so that a stream frozen
    to the floor scores far above a slow-but-moving stream. `floor` is set at fit time to a
    small fraction of the nominal median motion, which is what a truly frozen stream reads.
    Nominal pauses (a grasp, a wait) live inside the calibrated envelope because the conformal
    threshold is taken over nominal episodes that contain them. The first step of an episode
    scores 0 (no evidence).
    """

    kind = "stagnation"
    needs = "action_chunk"

    def __init__(self, window: int = 25, floor_fraction: float = 1e-3) -> None:
        super().__init__()
        self.window = int(window)
        self.floor_fraction = float(floor_fraction)
        self.log_mean: Optional[float] = None
        self.log_std: Optional[float] = None
        self.floor: Optional[float] = None
        self._prev: Optional[np.ndarray] = None
        self._recent: list[float] = []

    def reset(self) -> None:
        self._prev = None
        self._recent = []

    @staticmethod
    def _summary(chunk: np.ndarray) -> np.ndarray:
        c = np.asarray(chunk, dtype=np.float64)
        if c.ndim != 2:
            raise ValueError("expected a chunk [chunk, D]")
        return c.mean(axis=0)

    def raw_sequence(self, chunks: np.ndarray) -> np.ndarray:
        """Windowed mean motion per step for one episode [T, chunk, D]; step 0 is NaN."""
        chunks = np.asarray(chunks, dtype=np.float64)
        if chunks.ndim != 3:
            raise ValueError("expected chunks [T, chunk, D]")
        means = chunks.mean(axis=1)
        step = np.linalg.norm(np.diff(means, axis=0), axis=1)
        out = np.full(chunks.shape[0], np.nan, dtype=np.float64)
        for i in range(step.size):
            lo = max(0, i - self.window + 1)
            out[i + 1] = float(step[lo: i + 1].mean())
        return out

    def fit(self, nominal) -> "StagnationScorer":
        """nominal: one array [T, chunk, D] or a list of them (one per episode)."""
        seqs = [nominal] if isinstance(nominal, np.ndarray) and nominal.ndim == 3 else list(nominal)
        raws = np.concatenate([self.raw_sequence(np.asarray(s))[1:] for s in seqs if len(s) > 1])
        raws = raws[np.isfinite(raws)]
        if raws.size < 2:
            raise ValueError("need at least two consecutive chunks to fit")
        self.floor = float(max(np.median(raws) * self.floor_fraction, _EPS))
        logs = np.log(raws + self.floor)
        self.log_mean = float(logs.mean())
        self.log_std = float(max(logs.std(ddof=1), _EPS))
        self.fitted = True
        self.reset()
        return self

    def _score_raw(self, raw: float) -> float:
        z = (self.log_mean - np.log(raw + self.floor)) / self.log_std
        return float(max(0.0, z))

    def score(self, x: np.ndarray) -> float:
        if not self.fitted:
            raise RuntimeError("StagnationScorer.fit() first")
        cur = self._summary(x)
        if self._prev is None:
            self._prev = cur
            return 0.0
        self._recent.append(float(np.linalg.norm(cur - self._prev)))
        if len(self._recent) > self.window:
            self._recent = self._recent[-self.window:]
        self._prev = cur
        return self._score_raw(float(np.mean(self._recent)))

    def score_episode(self, episode: dict[str, Any]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("StagnationScorer.fit() first")
        acts = episode.get("actions")
        if acts is None:
            raise ValueError("episode has no 'actions' array, which stagnation needs")
        raw = self.raw_sequence(np.asarray(acts, dtype=np.float64))
        out = np.zeros(raw.shape[0], dtype=np.float64)
        for i in range(1, raw.shape[0]):
            out[i] = self._score_raw(float(raw[i]))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "window": self.window, "floor_fraction": self.floor_fraction,
                "log_mean": self.log_mean, "log_std": self.log_std, "floor": self.floor}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StagnationScorer":
        s = cls(window=d.get("window", 25), floor_fraction=d.get("floor_fraction", 1e-3))
        if d.get("log_mean") is not None:
            s.log_mean, s.log_std = float(d["log_mean"]), float(d["log_std"])
            s.floor, s.fitted = float(d["floor"]), True
        return s


class CompositeScorer(Scorer):
    """Weighted sum of standardised component scores.

    fit(nominal_episodes) takes a list of episode dicts, each with 'features' [T, F] and/or
    'actions' [T, chunk, D]. Each component is fitted on the data it needs, then every
    component score is standardised (z-scored) against its distribution over the nominal
    episodes so that a distance in chi-squared units and a distance in z units can be
    added. Weights default to equal. At score time, components whose input is missing are
    dropped and the weights renormalised over the ones present.
    """

    kind = "composite"
    needs = "any"

    def __init__(self, components: Optional[dict[str, Scorer]] = None,
                 weights: Optional[dict[str, float]] = None) -> None:
        super().__init__()
        self.components: dict[str, Scorer] = components or {}
        self.weights: dict[str, float] = weights or {k: 1.0 for k in self.components}
        self.comp_mean: dict[str, float] = {}
        self.comp_std: dict[str, float] = {}

    @classmethod
    def default(cls, has_features: bool = True, has_actions: bool = True,
                stride: Optional[int] = None, device: str = "auto",
                stagnation_window: int = 25) -> "CompositeScorer":
        comps: dict[str, Scorer] = {}
        if has_features:
            comps["mahalanobis"] = MahalanobisScorer(device=device)
        if has_actions:
            comps["chunk_consistency"] = ChunkConsistencyScorer(stride=stride)
            comps["action_stats"] = ActionStatsScorer()
            comps["stagnation"] = StagnationScorer(window=stagnation_window)
        if not comps:
            raise ValueError("a composite needs features or actions")
        return cls(comps)

    def reset(self) -> None:
        for c in self.components.values():
            c.reset()

    def fit(self, nominal: list[dict[str, Any]]) -> "CompositeScorer":
        episodes = list(nominal)
        if not episodes:
            raise ValueError("no nominal episodes")
        for name, comp in self.components.items():
            if comp.needs == "features":
                data = [np.asarray(e["features"], dtype=np.float64) for e in episodes if e.get("features") is not None]
                if not data:
                    raise ValueError(f"component {name} needs 'features' but no episode has them")
                comp.fit(np.concatenate(data, axis=0))
            else:
                data = [np.asarray(e["actions"], dtype=np.float64) for e in episodes if e.get("actions") is not None]
                if not data:
                    raise ValueError(f"component {name} needs 'actions' but no episode has them")
                if isinstance(comp, (ChunkConsistencyScorer, StagnationScorer)):
                    comp.fit(data)
                else:
                    comp.fit(np.concatenate(data, axis=0))
        for name, comp in self.components.items():
            raws = np.concatenate([comp.score_episode(e) for e in episodes])
            self.comp_mean[name] = float(raws.mean())
            self.comp_std[name] = float(max(raws.std(ddof=1), _EPS))
        self.fitted = True
        self.reset()
        return self

    def score_step(self, features: Optional[np.ndarray] = None,
                   action_chunk: Optional[np.ndarray] = None) -> tuple[Optional[float], dict[str, float]]:
        if not self.fitted:
            raise RuntimeError("CompositeScorer.fit() first")
        total, wsum, comps = 0.0, 0.0, {}
        for name, comp in self.components.items():
            s, _ = comp.score_step(features=features, action_chunk=action_chunk)
            if s is None:
                continue
            z = (s - self.comp_mean[name]) / self.comp_std[name]
            comps[name] = float(z)
            w = float(self.weights.get(name, 1.0))
            total += w * z
            wsum += w
        if wsum <= 0:
            return None, comps
        return total / wsum, comps

    def score(self, x: dict[str, Any]) -> float:
        s, _ = self.score_step(features=x.get("features"), action_chunk=x.get("action_chunk"))
        if s is None:
            raise ValueError("no usable input in x")
        return s

    def score_episode(self, episode: dict[str, Any]) -> np.ndarray:
        self.reset()
        total, wsum = None, 0.0
        for name, comp in self.components.items():
            key = "features" if comp.needs == "features" else "actions"
            if episode.get(key) is None:
                continue
            z = (comp.score_episode(episode) - self.comp_mean[name]) / self.comp_std[name]
            w = float(self.weights.get(name, 1.0))
            total = w * z if total is None else total + w * z
            wsum += w
        if total is None:
            raise ValueError("episode has neither features nor actions")
        return total / wsum

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "components": {k: c.to_dict() for k, c in self.components.items()},
                "weights": self.weights, "comp_mean": self.comp_mean, "comp_std": self.comp_std}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CompositeScorer":
        comps = {k: scorer_from_dict(v) for k, v in d["components"].items()}
        s = cls(comps, dict(d.get("weights", {})))
        s.comp_mean = {k: float(v) for k, v in d.get("comp_mean", {}).items()}
        s.comp_std = {k: float(v) for k, v in d.get("comp_std", {}).items()}
        s.fitted = bool(s.comp_mean)
        return s


_REGISTRY: dict[str, type[Scorer]] = {
    MahalanobisScorer.kind: MahalanobisScorer,
    ChunkConsistencyScorer.kind: ChunkConsistencyScorer,
    ActionStatsScorer.kind: ActionStatsScorer,
    StagnationScorer.kind: StagnationScorer,
    CompositeScorer.kind: CompositeScorer,
}


def scorer_from_dict(d: dict[str, Any]) -> Scorer:
    """Rebuild any scorer from its to_dict() output."""
    kind = d.get("type")
    if kind not in _REGISTRY:
        raise ValueError(f"unknown scorer type {kind!r}")
    return _REGISTRY[kind].from_dict(d)

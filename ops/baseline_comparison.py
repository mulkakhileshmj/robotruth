"""Do published runtime failure detectors see real task failures from the action stream?

Our guard is near chance at spotting real task failures in recorded robot logs while
catching injected execution faults almost perfectly. That could be a fact about our
scorers or a fact about the signal they read. This experiment separates the two by running
the published families of runtime failure signals over exactly the same episodes.

The baselines are action-stream implementations of the signal families used in the
literature:

  chunk_consistency  Action Chunk Consistency, the cheap half of VLA-FAIL (Seligmann et al.,
                     arXiv 2606.21386): consecutive receding-horizon chunks overlap, so
                     disagreement over the overlap flags trouble.
  mahalanobis        Distance from the nominal feature distribution, the post-hoc embedding
                     distance of FAIL-Detect (Xu et al., RSS 2025) and the action-side
                     analogue of VLA-FAIL's Last-Layer Mahalanobis Distance, which reads
                     vision-language features instead.
  action_variance    Spread of the action chunk, FAIL-Detect's post-hoc action-sample
                     variance signal.
  rnd                Random Network Distillation (Burda et al., ICLR 2019), the learned
                     novelty signal FAIL-Detect evaluates.
  density            Log density under a Gaussian mixture fitted to nominal features,
                     standing in for FAIL-Detect's flow-based density estimator, which was
                     its most reliable signal.
  pca_recon          Reconstruction error from a PCA basis fitted on nominal features.
  iforest            Isolation Forest (Liu et al., ICDM 2008).
  ocsvm              One-Class SVM (Schoelkopf et al., 2001).
  lstm_pred          Next-step prediction error from a GRU trained on nominal action
                     streams, the classic sequence anomaly detector (Malhotra et al., 2016).
  ours               The robotruth composite plus its separate stagnation head.

Every method is fitted on successful episodes only, which is the setting all of these
papers work in, and scored two ways on held-out data: against the corpus's own real task
failures, and against execution faults injected into held-out successes. AUROC is used
because it needs no threshold, so a low number cannot be blamed on calibration.

Usage: python ops/baseline_comparison.py <data_root> <out_dir> [--max-episodes 400]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_real_faults import inject  # noqa: E402
from guard_real_logs import collect  # noqa: E402

warnings.filterwarnings("ignore")

FAULTS = (("freeze", 1.0), ("offset", 2.0), ("noise", 0.5))


# ---------------------------------------------------------------------------- utilities

def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a failure > score of a success), ties at half. 0.5 is a coin flip."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    uniq, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    order = allv.argsort()
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def auroc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 400, seed: int = 0) -> tuple:
    """Point estimate with a bootstrap interval, because an AUROC without one is a vibe."""
    a = auroc(pos, neg)
    if not np.isfinite(a):
        return a, None, None
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    vals = []
    for _ in range(n_boot):
        p = pos[rng.integers(0, pos.size, pos.size)]
        n = neg[rng.integers(0, neg.size, neg.size)]
        v = auroc(p, n)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return a, None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return a, float(lo), float(hi)


def episode_features(ep: dict) -> np.ndarray:
    """Per-step feature rows: [state, action] when the state is available, else the action."""
    if ep.get("features") is not None:
        return np.asarray(ep["features"], dtype=np.float64)
    ch = np.asarray(ep["actions"], dtype=np.float64)
    return ch.mean(axis=1)


def chunk_stream(ep: dict) -> np.ndarray:
    return np.asarray(ep["actions"], dtype=np.float64)


# ------------------------------------------------------------------------ the detectors

class Detector:
    """Fit on nominal episodes, then produce one scalar per episode.

    Every detector reduces an episode to its peak step score, which is how a runtime
    monitor behaves: it alarms if any step crosses, so the episode-level statistic is the
    maximum over steps.
    """

    name = "base"

    def fit(self, eps: list[dict]) -> "Detector":
        raise NotImplementedError

    def step_scores(self, ep: dict) -> np.ndarray:
        raise NotImplementedError

    def score(self, ep: dict) -> float:
        v = np.asarray(self.step_scores(ep), dtype=np.float64)
        v = v[np.isfinite(v)]
        return float(v.max()) if v.size else float("nan")


class ChunkConsistency(Detector):
    name = "chunk_consistency"

    def fit(self, eps):
        return self

    def step_scores(self, ep):
        ch = chunk_stream(ep)
        if ch.ndim != 3 or ch.shape[0] < 2:
            return np.zeros(1)
        k = ch.shape[1]
        stride = max(1, k // 2)
        out = np.zeros(ch.shape[0])
        for i in range(1, ch.shape[0]):
            a, b = ch[i - 1][stride:], ch[i][: k - stride]
            out[i] = np.mean(np.linalg.norm(a - b, axis=1)) if a.shape[0] else 0.0
        return out


class MahalanobisDet(Detector):
    name = "mahalanobis"

    def fit(self, eps):
        x = np.concatenate([episode_features(e) for e in eps], axis=0)
        self.mu = x.mean(axis=0)
        cov = np.cov(x, rowvar=False) + np.eye(x.shape[1]) * 1e-6
        self.prec = np.linalg.pinv(cov)
        return self

    def step_scores(self, ep):
        d = episode_features(ep) - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


class ActionVariance(Detector):
    name = "action_variance"

    def fit(self, eps):
        v = np.concatenate([self._raw(e) for e in eps])
        self.mu, self.sd = float(v.mean()), float(v.std() + 1e-9)
        return self

    @staticmethod
    def _raw(ep):
        ch = chunk_stream(ep)
        return ch.var(axis=1).mean(axis=1) if ch.ndim == 3 else np.zeros(len(ch))

    def step_scores(self, ep):
        return np.abs((self._raw(ep) - self.mu) / self.sd)


class PCARecon(Detector):
    name = "pca_recon"

    def fit(self, eps):
        from sklearn.decomposition import PCA
        x = np.concatenate([episode_features(e) for e in eps], axis=0)
        k = max(1, min(8, x.shape[1] - 1))
        self.p = PCA(n_components=k).fit(x)
        return self

    def step_scores(self, ep):
        x = episode_features(ep)
        return np.sum((x - self.p.inverse_transform(self.p.transform(x))) ** 2, axis=1)


class IForest(Detector):
    name = "iforest"

    def fit(self, eps):
        from sklearn.ensemble import IsolationForest
        x = np.concatenate([episode_features(e) for e in eps], axis=0)
        idx = np.random.default_rng(0).choice(x.shape[0], min(20000, x.shape[0]), replace=False)
        self.m = IsolationForest(n_estimators=100, random_state=0).fit(x[idx])
        return self

    def step_scores(self, ep):
        return -self.m.score_samples(episode_features(ep))


class OCSVM(Detector):
    name = "ocsvm"

    def fit(self, eps):
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
        x = np.concatenate([episode_features(e) for e in eps], axis=0)
        idx = np.random.default_rng(0).choice(x.shape[0], min(4000, x.shape[0]), replace=False)
        self.sc = StandardScaler().fit(x[idx])
        self.m = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(self.sc.transform(x[idx]))
        return self

    def step_scores(self, ep):
        return -self.m.score_samples(self.sc.transform(episode_features(ep)))


class GMMDensity(Detector):
    name = "density"

    def fit(self, eps):
        from sklearn.mixture import GaussianMixture
        x = np.concatenate([episode_features(e) for e in eps], axis=0)
        idx = np.random.default_rng(0).choice(x.shape[0], min(20000, x.shape[0]), replace=False)
        k = 5 if x.shape[0] > 500 else 2
        self.m = GaussianMixture(n_components=k, covariance_type="diag", random_state=0).fit(x[idx])
        return self

    def step_scores(self, ep):
        return -self.m.score_samples(episode_features(ep))


class RND(Detector):
    """Random Network Distillation: a trained net predicts a frozen random net's output.

    Error is high where nominal data was sparse, which is the learned novelty signal
    FAIL-Detect evaluates alongside its flow density.
    """

    name = "rnd"

    def fit(self, eps):
        import torch
        import torch.nn as nn
        x = np.concatenate([episode_features(e) for e in eps], axis=0).astype(np.float32)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        xs = torch.tensor((x - self.mu) / self.sd)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        d = xs.shape[1]
        torch.manual_seed(0)
        self.target = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 32)).to(dev).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.pred = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                  nn.Linear(64, 32)).to(dev)
        opt = torch.optim.Adam(self.pred.parameters(), lr=1e-3)
        xs = xs.to(dev)
        n = xs.shape[0]
        for _ in range(300):
            i = torch.randint(0, n, (min(512, n),), device=dev)
            loss = ((self.pred(xs[i]) - self.target(xs[i])) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        self.pred.eval()
        self.torch, self.dev = torch, dev
        return self

    def step_scores(self, ep):
        torch = self.torch
        x = ((episode_features(ep).astype(np.float32) - self.mu) / self.sd)
        with torch.no_grad():
            t = torch.tensor(x, device=self.dev)
            return ((self.pred(t) - self.target(t)) ** 2).mean(dim=1).cpu().numpy()


class GRUPredict(Detector):
    """Next-step prediction error from a GRU trained only on nominal action streams."""

    name = "lstm_pred"

    def fit(self, eps):
        import torch
        import torch.nn as nn
        seqs = [episode_features(e).astype(np.float32) for e in eps]
        x = np.concatenate(seqs, axis=0)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        d = x.shape[1]
        torch.manual_seed(0)
        self.net = nn.GRU(d, 64, batch_first=True).to(dev)
        self.head = nn.Linear(64, d).to(dev)
        opt = torch.optim.Adam(list(self.net.parameters()) + list(self.head.parameters()), lr=1e-3)
        win = 32
        chunks = []
        for s in seqs:
            s = (s - self.mu) / self.sd
            for i in range(0, max(1, len(s) - win), win):
                w = s[i: i + win + 1]
                if len(w) == win + 1:
                    chunks.append(w)
        if not chunks:
            self.torch, self.dev = torch, dev
            return self
        data = torch.tensor(np.stack(chunks), device=dev)
        for _ in range(200):
            i = torch.randint(0, data.shape[0], (min(128, data.shape[0]),), device=dev)
            b = data[i]
            out, _ = self.net(b[:, :-1])
            loss = ((self.head(out) - b[:, 1:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        self.net.eval(); self.head.eval()
        self.torch, self.dev = torch, dev
        return self

    def step_scores(self, ep):
        torch = self.torch
        s = (episode_features(ep).astype(np.float32) - self.mu) / self.sd
        if len(s) < 3:
            return np.zeros(1)
        with torch.no_grad():
            t = torch.tensor(s[None], device=self.dev)
            out, _ = self.net(t[:, :-1])
            err = ((self.head(out) - t[:, 1:]) ** 2).mean(dim=2)[0].cpu().numpy()
        return np.concatenate([[0.0], err])


class Ours(Detector):
    name = "ours"

    def fit(self, eps):
        from robotruth.guard import calibrate_guard
        strip = lambda e: {k: e[k] for k in ("features", "actions", "timestamps")
                           if e.get(k) is not None}
        self.guard, _ = calibrate_guard([strip(e) for e in eps], alpha=0.05, method="max",
                                        dt=0.05, stride=1)
        self.strip = strip
        return self

    def step_scores(self, ep):
        return np.asarray(self.guard.scorer.score_episode(self.strip(ep)), dtype=np.float64)


DETECTORS = [ChunkConsistency, MahalanobisDet, ActionVariance, GMMDensity, PCARecon,
             IForest, OCSVM, RND, GRUPredict, Ours]


# ----------------------------------------------------------------------------- the run

def faulted(ep: dict, fault: str, mag: float, rng) -> dict:
    ch = np.asarray(ep["actions"], dtype=np.float64)
    onset = ch.shape[0] // 2
    out = dict(ep)
    out["actions"] = inject(ch, fault, onset, rng, mag)
    if ep.get("features") is not None:
        f = np.array(ep["features"], dtype=np.float64, copy=True)
        n_act = ch.shape[-1]
        f[:, -n_act:] = inject(f[:, -n_act:][:, None, :], fault, onset, rng, mag)[:, 0, :]
        out["features"] = f
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("out")
    ap.add_argument("--max-episodes", type=int, default=400)
    args = ap.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sources = collect(root, args.max_episodes, 10)
    results: dict = {}
    t0 = time.time()

    for name, data in sorted(sources.items()):
        eps = data["episodes"]
        succ = [e for e in eps if e["success"]]
        fail = [e for e in eps if not e["success"]]
        if len(succ) < 60 or len(fail) < 20:
            continue
        rng = np.random.default_rng(0)
        order = rng.permutation(len(succ))
        n_fit = int(0.6 * len(succ))
        fit_eps = [succ[i] for i in order[:n_fit]]
        held = [succ[i] for i in order[n_fit:]]
        print(f"\n=== {name}: fit {len(fit_eps)}, held-out successes {len(held)}, "
              f"real failures {len(fail)}", flush=True)

        frng = np.random.default_rng(5)
        fault_sets = {f"{f} {m:g} sd" if f != "freeze" else "freeze":
                      [faulted(e, f, m, frng) for e in held] for f, m in FAULTS}

        rows = {}
        for cls in DETECTORS:
            det = cls()
            try:
                t1 = time.time()
                det.fit(fit_eps)
                s_held = np.array([det.score(e) for e in held])
                s_fail = np.array([det.score(e) for e in fail])
                a, lo, hi = auroc_ci(s_fail, s_held)
                row = {"real_task_failures": {"auroc": a, "lo": lo, "hi": hi,
                                              "n_fail": len(fail), "n_ok": len(held)}}
                for label, fset in fault_sets.items():
                    s_f = np.array([det.score(e) for e in fset])
                    fa, flo, fhi = auroc_ci(s_f, s_held)
                    row[label] = {"auroc": fa, "lo": flo, "hi": fhi}
                row["fit_seconds"] = round(time.time() - t1, 1)
                rows[cls.name] = row
                msg = " ".join(f"{k.split()[0]}={v['auroc']:.2f}" for k, v in row.items()
                               if isinstance(v, dict))
                print(f"  {cls.name:18s} {msg}", flush=True)
            except Exception as e:  # noqa: BLE001
                rows[cls.name] = {"error": str(e)[:200]}
                print(f"  {cls.name:18s} FAILED {str(e)[:120]}", flush=True)
        results[name] = {"robot": data["robot"], "fps": data["fps"], "n_fit": len(fit_eps),
                         "n_held": len(held), "n_fail": len(fail), "detectors": rows}

    (out / "baseline_comparison.json").write_text(json.dumps(results, indent=1, default=str),
                                                  encoding="utf-8")

    lines = ["# Do published failure signals see real task failures in the action stream?", "",
             "AUROC of each detector's per-episode peak score, real failures against held-out "
             "successes, and the same detectors against execution faults injected into those "
             "same held-out successes. Every detector is fitted on successful episodes only. "
             "0.5 is a coin flip, and AUROC needs no threshold, so a low number here cannot be "
             "explained by bad calibration.", ""]
    for name, r in results.items():
        lines += [f"## {name}", "",
                  f"Robot `{r['robot']}`, {r['fps']:g} Hz. Fitted on {r['n_fit']} successes, "
                  f"evaluated on {r['n_held']} held-out successes and {r['n_fail']} real failures.",
                  "", "| detector | real task failures | " +
                  " | ".join(k for k in next(iter(r["detectors"].values())) if k not in
                             ("real_task_failures", "fit_seconds", "error")) + " |",
                  "|---|---|" + "---|" * 3]
        for det, row in r["detectors"].items():
            if "error" in row:
                lines.append(f"| {det} | error | | | |")
                continue
            cells = []
            for k, v in row.items():
                if k == "fit_seconds" or not isinstance(v, dict):
                    continue
                cells.append(f"{v['auroc']:.3f}" + (f" [{v['lo']:.2f}, {v['hi']:.2f}]"
                                                    if v.get("lo") is not None else ""))
            lines.append(f"| {det} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"Total wall time {time.time()-t0:.0f} s.", "", "BASELINE_COMPARISON_COMPLETE"]
    (out / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nBASELINE_COMPARISON_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

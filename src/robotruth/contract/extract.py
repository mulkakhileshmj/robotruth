"""Adapters that build an ExecSpec from a checkpoint.

Adapters never import the policy framework. They read files: config.json, train_config.json,
policy_preprocessor.json, safetensors headers, norm-stats json. Anything they cannot read
becomes `unknown` and is listed in `spec.unverified`, so the checker can fail closed.

Two kinds of source:
- a real checkpoint directory (weights present, hashed here);
- a probe directory produced by ops/remote_checkpoint_probe.sh on a GPU box: `files.json`
  (name, size, sha256 for every file), `configs/` (small JSON files verbatim) and `headers/`
  (safetensors headers). This lets the manifest be built without moving weights.

Families and what was learned from real checkpoints (Sep 2026 probe of five public repos):
- lerobot (SmolVLA, pi0, pi05, ACT): `config.json` has type, chunk_size, n_action_steps,
  normalization_mapping, input and output features, resize settings. Newer checkpoints carry
  `policy_preprocessor.json` whose normalizer step may have a `state_file` with the
  statistics (SmolVLA ships stats for several robots in one file, keyed by dataset name);
  base pi0 and pi05 ship NO statistics, so the stats hash is unknown until fine-tuning. Older
  checkpoints (ACT) keep `normalize_*` buffers inside model.safetensors. `train_config.json`,
  when present, gives dataset fps and env type.
- gr00t: `config.json` has action_horizon and action_dim (padded max); `experiment_cfg/
  metadata.json` has per-embodiment statistics, modality shapes, absolute flags and video fps.
  Which embodiment tag is used at runtime selects the statistics: record it in the overlay.
- openpi: `assets/<id>/norm_stats.json` and orbax `params/`; horizon and cameras live in code.
- custom: hash everything, take the rest from an overlay.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from robotruth.contract.spec import (
    CameraSpec,
    ExecSpec,
    sha256_file,
    sha256_obj,
)

WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".msgpack", ".npz"}


# --------------------------------------------------------------------------------------
# Source abstraction
# --------------------------------------------------------------------------------------
@dataclass
class FileInfo:
    name: str
    size: int
    sha256: str


class Source:
    """Uniform read access to a real checkpoint directory or a probe directory."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.is_probe = (self.root / "files.json").exists() and (self.root / "configs").exists()
        self._files: Optional[list[FileInfo]] = None

    # listing -------------------------------------------------------------------------
    def files(self) -> list[FileInfo]:
        if self._files is None:
            if self.is_probe:
                data = json.loads((self.root / "files.json").read_text(encoding="utf-8"))
                self._files = [FileInfo(f["name"].replace("\\", "/"), int(f["size"]), f["sha256"]) for f in data["files"]]
            else:
                out = []
                for p in sorted(self.root.rglob("*")):
                    if p.is_file():
                        out.append(FileInfo(str(p.relative_to(self.root)).replace("\\", "/"), p.stat().st_size, sha256_file(p)))
                self._files = out
        return self._files

    def exists(self, rel: str) -> bool:
        return any(f.name == rel for f in self.files())

    def names(self) -> list[str]:
        return [f.name for f in self.files()]

    def combined_hash(self, rels: list[str]) -> Optional[str]:
        infos = {f.name: f for f in self.files()}
        chosen = sorted(r for r in rels if r in infos)
        if not chosen:
            return None
        import hashlib
        h = hashlib.sha256()
        for r in chosen:
            h.update(Path(r).name.encode())
            h.update(bytes.fromhex(infos[r].sha256))
        return h.hexdigest()

    # content -------------------------------------------------------------------------
    def read_json(self, rel: str) -> Optional[dict]:
        p = (self.root / "configs" / rel) if self.is_probe else (self.root / rel)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def header(self, rel: str) -> Optional[dict]:
        """safetensors header (tensor names, dtypes, shapes) without reading tensor data."""
        if self.is_probe:
            p = self.root / "headers" / (rel.replace("/", "__") + ".header.json")
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        p = self.root / rel
        try:
            with p.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                return json.loads(f.read(n).decode("utf-8"))
        except Exception:
            return None

    def weight_files(self) -> list[FileInfo]:
        return [f for f in self.files() if Path(f.name).suffix.lower() in WEIGHT_SUFFIXES
                and "preprocessor" not in f.name and "postprocessor" not in f.name]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _norm_type(mapping: Optional[dict], key: str) -> str:
    if not mapping:
        return "unknown"
    v = str(mapping.get(key, "")).upper()
    if "MEAN" in v:
        return "mean_std"
    if "MIN" in v:
        return "min_max"
    if "QUANT" in v:
        return "quantile"
    if "IDENT" in v:
        return "identity"
    return "unknown"


def _set_weights(src: Source, spec: ExecSpec) -> None:
    w = src.weight_files()
    if w:
        spec.policy.weights_files = [f.name for f in w]
        spec.policy.weights_sha256 = src.combined_hash([f.name for f in w])
        spec.policy.weights_bytes = sum(f.size for f in w)
    else:
        spec.unverified.append("policy.weights_sha256")


def _mark_unknown(spec: ExecSpec, pairs: list[tuple[str, Any]]) -> None:
    for f, v in pairs:
        if v is None or v == "unknown":
            spec.unverified.append(f)


# --------------------------------------------------------------------------------------
# lerobot
# --------------------------------------------------------------------------------------
def _lerobot(src: Source, spec: ExecSpec) -> ExecSpec:
    cfg = src.read_json("config.json") or {}
    train_cfg = src.read_json("train_config.json") or {}
    pre = src.read_json("policy_preprocessor.json") or {}
    post = src.read_json("policy_postprocessor.json") or {}

    spec.policy.family = "lerobot"
    spec.policy.name = str(cfg.get("type") or (train_cfg.get("policy") or {}).get("type") or src.root.name)
    _set_weights(src, spec)
    if src.exists("config.json"):
        spec.policy.config_sha256 = src.combined_hash(["config.json"])

    out_feats = cfg.get("output_features") or {}
    in_feats = cfg.get("input_features") or {}
    shape = (out_feats.get("action") or {}).get("shape")
    spec.action.dim = int(shape[0]) if shape else None
    spec.action.chunk_size = cfg.get("chunk_size") or cfg.get("horizon") or cfg.get("action_horizon")
    spec.action.n_action_steps = cfg.get("n_action_steps")
    fps = (train_cfg.get("dataset") or {}).get("fps") or (train_cfg.get("env") or {}).get("fps") or cfg.get("fps")
    spec.action.control_hz = float(fps) if fps else None

    nm = cfg.get("normalization_mapping") or {}
    if not nm:  # pi05 style: mapping lives only in the preprocessor
        for step in pre.get("steps", []):
            if "normalizer" in step.get("registry_name", ""):
                nm = step.get("config", {}).get("norm_map") or {}
    spec.action.normalization.type = _norm_type(nm, "ACTION")
    spec.observation.normalization.type = _norm_type(nm, "STATE")

    # Normalization statistics: preprocessor state file (new), stats.json, or buffers in the weights (old).
    stats_rels: list[str] = []
    for proc in (pre, post):
        for step in proc.get("steps", []):
            sf = step.get("state_file")
            if sf and src.exists(sf):
                stats_rels.append(sf)
    for rel in src.names():
        if Path(rel).name in ("stats.json", "norm_stats.json", "dataset_stats.json"):
            stats_rels.append(rel)
    stats_hash: Optional[str] = None
    stats_src: Optional[str] = None
    if stats_rels:
        stats_hash = src.combined_hash(sorted(set(stats_rels)))
        stats_src = ",".join(sorted(set(stats_rels)))
        hdr = src.header(stats_rels[0]) if stats_rels[0].endswith(".safetensors") else None
        if hdr:
            datasets = sorted({k.split(".buffer.")[0] for k in hdr if ".buffer." in k})
            if len(datasets) > 1:
                spec.notes.append(f"normalizer file holds statistics for {len(datasets)} datasets ({', '.join(datasets)}); "
                                  "the one selected at runtime is part of the executable policy. Record it in the overlay as action.normalization.stats_source.")
    else:
        w = src.weight_files()
        hdr = src.header(w[0].name) if w and w[0].name.endswith(".safetensors") else None
        if hdr:
            keys = sorted(k for k in hdr if k.startswith("normalize_") or k.startswith("unnormalize_"))
            if keys:
                stats_hash = sha256_obj({k: hdr[k] for k in keys})
                stats_src = "weights:normalize_* buffers (metadata hash; values are inside weights hash)"
        if stats_hash is None and (pre or cfg):
            spec.notes.append("no normalization statistics shipped with this checkpoint (base model); stats are set at fine-tune time and must be recorded then.")
    spec.action.normalization.stats_sha256 = stats_hash
    spec.action.normalization.stats_source = stats_src
    spec.action.normalization.action_dim = spec.action.dim
    spec.observation.normalization.stats_sha256 = stats_hash
    spec.observation.normalization.stats_source = stats_src

    if cfg.get("use_delta_joint_actions_aloha") is True or cfg.get("use_delta_joint_actions") is True or cfg.get("relative_actions") is True:
        spec.action.mode = "delta"
    elif cfg:
        spec.action.mode = "absolute"
    spec.action.space = "joint_position" if cfg else "unknown"

    for key, feat in in_feats.items():
        if str(feat.get("type", "")).upper() == "VISUAL" or "image" in key:
            sh = feat.get("shape") or [None, None, None]
            spec.observation.cameras.append(CameraSpec(name=key, channels=sh[0], height=sh[1], width=sh[2]))
        elif str(feat.get("type", "")).upper() == "STATE" or "state" in key:
            spec.observation.state_keys.append(key)
            sh = feat.get("shape")
            spec.observation.state_dim = int(sh[0]) if sh else None
    resize = cfg.get("resize_imgs_with_padding") or cfg.get("image_resolution") or cfg.get("image_resize") or cfg.get("crop_shape")
    spec.observation.image_resize = str(resize) if resize else None
    tokenizer = None
    rename_map = None
    for step in pre.get("steps", []):
        c = step.get("config", {})
        if "tokenizer" in step.get("registry_name", ""):
            tokenizer = {k: c.get(k) for k in ("tokenizer_name", "max_length", "padding", "padding_side", "truncation")}
        if "rename" in step.get("registry_name", ""):
            rename_map = c.get("rename_map")
    if cfg or pre:
        spec.observation.preprocessing_sha256 = sha256_obj({
            "resize": resize, "crop_is_random": cfg.get("crop_is_random"), "normalization_mapping": nm,
            "input_features": in_feats, "n_obs_steps": cfg.get("n_obs_steps"), "tokenizer": tokenizer,
            "rename_map": rename_map, "empty_cameras": cfg.get("empty_cameras"), "adapt_to_pi_aloha": cfg.get("adapt_to_pi_aloha"),
            "steps": [s.get("registry_name") for s in pre.get("steps", [])],
        })
    spec.runtime.framework = "lerobot"
    spec.runtime.dtype = cfg.get("dtype") or ("bfloat16" if cfg.get("use_amp") else None)
    spec.runtime.device = cfg.get("device")
    env_type = (train_cfg.get("env") or {}).get("type")
    repo = (train_cfg.get("dataset") or {}).get("repo_id")
    spec.embodiment.robot = str(env_type or repo or "unknown")
    if cfg.get("num_inference_steps") or cfg.get("num_steps"):
        spec.notes.append(f"denoising steps: {cfg.get('num_inference_steps') or cfg.get('num_steps')} (drives latency; VLA-Perf)")

    _mark_unknown(spec, [
        ("action.dim", spec.action.dim), ("action.chunk_size", spec.action.chunk_size),
        ("action.n_action_steps", spec.action.n_action_steps), ("action.control_hz", spec.action.control_hz),
        ("action.normalization.stats_sha256", stats_hash), ("observation.state_dim", spec.observation.state_dim),
        ("embodiment.robot", None if spec.embodiment.robot == "unknown" else spec.embodiment.robot),
    ])
    spec.unverified.append("action.gripper_convention")
    return spec


# --------------------------------------------------------------------------------------
# gr00t
# --------------------------------------------------------------------------------------
def _gr00t(src: Source, spec: ExecSpec, embodiment_tag: Optional[str] = None) -> ExecSpec:
    spec.policy.family = "gr00t"
    cfg = src.read_json("config.json") or {}
    meta = src.read_json("experiment_cfg/metadata.json") or {}
    spec.policy.name = str(cfg.get("model_type") or src.root.name)
    _set_weights(src, spec)
    if src.exists("config.json"):
        spec.policy.config_sha256 = src.combined_hash(["config.json"])
    head = cfg.get("action_head_cfg") or {}
    spec.action.chunk_size = cfg.get("action_horizon") or head.get("action_horizon")
    spec.action.dim = cfg.get("action_dim") or head.get("action_dim")
    spec.runtime.dtype = cfg.get("torch_dtype") or cfg.get("model_dtype")
    if head.get("num_inference_timesteps"):
        spec.notes.append(f"denoising steps: {head.get('num_inference_timesteps')}")

    tags = sorted(meta.keys()) if isinstance(meta, dict) else []
    if tags:
        chosen = embodiment_tag if embodiment_tag in meta else None
        if chosen is None and len(tags) == 1:
            chosen = tags[0]
        if chosen:
            emb = meta[chosen]
            spec.embodiment.robot = str(emb.get("embodiment_tag") or chosen)
            spec.action.normalization.stats_sha256 = sha256_obj(emb.get("statistics"))
            spec.action.normalization.stats_source = f"experiment_cfg/metadata.json[{chosen}].statistics"
            spec.action.normalization.type = "quantile" if "q01" in json.dumps(emb.get("statistics", {}))[:5000] else "mean_std"
            mods = emb.get("modalities") or {}
            act = mods.get("action") or {}
            if act:
                spec.action.dim = int(sum(int((v.get("shape") or [0])[0]) for v in act.values()))
                spec.action.mode = "absolute" if all(v.get("absolute", True) for v in act.values()) else "delta"
                spec.action.space = "joint_position"
            st = mods.get("state") or {}
            if st:
                spec.observation.state_dim = int(sum(int((v.get("shape") or [0])[0]) for v in st.values()))
                spec.observation.state_keys = sorted(st.keys())
            vids = mods.get("video") or {}
            for name, v in vids.items():
                res = v.get("resolution") or [None, None]
                spec.observation.cameras.append(CameraSpec(name=name, width=res[0], height=res[1], channels=v.get("channels")))
                if v.get("fps"):
                    spec.action.control_hz = float(v["fps"])
            spec.observation.normalization.stats_sha256 = spec.action.normalization.stats_sha256
            spec.observation.normalization.stats_source = spec.action.normalization.stats_source
            spec.observation.normalization.type = spec.action.normalization.type
            spec.observation.preprocessing_sha256 = sha256_obj({"modalities": mods, "tag": chosen})
        else:
            spec.notes.append(f"metadata holds {len(tags)} embodiment tags ({', '.join(tags)}); pass embodiment_tag in the overlay to select the statistics that will run.")
            spec.unverified += ["action.normalization.stats_sha256", "embodiment.robot", "action.mode", "action.space",
                                "observation.preprocessing_sha256", "observation.state_dim", "action.control_hz"]
    else:
        spec.unverified += ["action.normalization.stats_sha256", "embodiment.robot", "action.mode", "action.space",
                            "observation.preprocessing_sha256", "observation.state_dim", "action.control_hz"]
    spec.runtime.framework = "gr00t"
    spec.action.n_action_steps = spec.action.n_action_steps or None
    spec.unverified += ["action.gripper_convention", "action.n_action_steps"]
    return spec


# --------------------------------------------------------------------------------------
# openpi
# --------------------------------------------------------------------------------------
def _openpi(src: Source, spec: ExecSpec) -> ExecSpec:
    spec.policy.family = "openpi"
    spec.policy.name = src.root.name
    weights = [f for f in src.files() if "params/" in f.name or Path(f.name).suffix in WEIGHT_SUFFIXES]
    if weights:
        spec.policy.weights_files = [f.name for f in weights[:50]]
        spec.policy.weights_sha256 = src.combined_hash([f.name for f in weights])
        spec.policy.weights_bytes = sum(f.size for f in weights)
    else:
        spec.unverified.append("policy.weights_sha256")
    stats = [n for n in src.names() if Path(n).name == "norm_stats.json"]
    if stats:
        spec.action.normalization.stats_sha256 = src.combined_hash(stats)
        spec.action.normalization.stats_source = ",".join(stats)
        st = src.read_json(stats[0]) or {}
        norm = st.get("norm_stats") or st
        act = norm.get("actions") or norm.get("action") or {}
        if "q01" in act or "q99" in act:
            spec.action.normalization.type = "quantile"
        elif "mean" in act:
            spec.action.normalization.type = "mean_std"
        mean = act.get("mean")
        spec.action.dim = len(mean) if isinstance(mean, list) else None
        spec.action.normalization.action_dim = spec.action.dim
        spec.observation.normalization.stats_sha256 = spec.action.normalization.stats_sha256
        spec.observation.normalization.type = spec.action.normalization.type
    else:
        spec.unverified.append("action.normalization.stats_sha256")
    spec.runtime.framework = "openpi"
    spec.notes.append("openpi: action horizon, control rate and cameras are defined in code (config.py); supply them via an overlay.")
    spec.unverified += ["action.chunk_size", "action.control_hz", "action.mode", "action.space",
                        "observation.preprocessing_sha256", "observation.state_dim", "action.gripper_convention", "embodiment.robot"]
    return spec


# --------------------------------------------------------------------------------------
# custom
# --------------------------------------------------------------------------------------
def _custom(src: Source, spec: ExecSpec) -> ExecSpec:
    spec.policy.family = "custom"
    spec.policy.name = src.root.name
    _set_weights(src, spec)
    spec.unverified += ["action.normalization.stats_sha256", "action.chunk_size", "action.control_hz", "action.mode",
                        "action.space", "action.dim", "observation.preprocessing_sha256", "observation.state_dim",
                        "action.gripper_convention", "action.n_action_steps", "embodiment.robot"]
    return spec


# --------------------------------------------------------------------------------------
# overlay and entry point
# --------------------------------------------------------------------------------------
def _apply_overlay(spec: ExecSpec, overlay: dict[str, Any]) -> ExecSpec:
    """Merge a user-supplied partial manifest (nested or dotted keys) and clear resolved unverified entries."""
    data = spec.model_dump()
    unverified: list[str] = list(data.get("unverified", []))

    def set_path(d: dict, parts: list[str], value: Any) -> None:
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value

    def walk(prefix: list[str], node: Any) -> None:
        if isinstance(node, dict) and not (prefix and prefix[-1] == "cameras"):
            for k, v in node.items():
                walk(prefix + k.split("."), v)
        else:
            set_path(data, prefix, node)
            dotted = ".".join(prefix)
            if dotted in unverified:
                unverified.remove(dotted)

    for k, v in overlay.items():
        if k in ("embodiment_tag",):
            continue
        walk(k.split("."), v)
    data["unverified"] = unverified
    return ExecSpec.model_validate(data)


def detect_family(src: Source) -> str:
    cfg = src.read_json("config.json")
    if cfg:
        if "action_horizon" in cfg or "gr00t" in str(cfg.get("model_type", "")).lower() or src.exists("experiment_cfg/metadata.json"):
            return "gr00t"
        if "input_features" in cfg or "output_features" in cfg or "normalization_mapping" in cfg or src.exists("policy_preprocessor.json"):
            return "lerobot"
    if any(Path(n).name == "norm_stats.json" for n in src.names()) or any(n.startswith("params/") for n in src.names()):
        return "openpi"
    return "custom"


def extract_spec(path: Path | str, family: Optional[str] = None, overlay: Optional[dict[str, Any]] = None,
                 role: str = "unspecified") -> ExecSpec:
    dir_ = Path(path)
    if not dir_.exists():
        raise FileNotFoundError(dir_)
    if dir_.is_file():
        dir_ = dir_.parent
    src = Source(dir_)
    family = family or detect_family(src)
    spec = ExecSpec(role=role)  # type: ignore[arg-type]
    spec.policy.source_path = str(dir_)
    if src.is_probe:
        spec.notes.append("built from a probe directory (hashes and configs); weights were hashed on the machine that holds them.")
    overlay = dict(overlay or {})
    overlay_file = dir_ / "robotruth.spec.json"
    if overlay_file.exists():
        overlay = {**(json.loads(overlay_file.read_text(encoding="utf-8")) or {}), **overlay}
    if family == "gr00t":
        spec = _gr00t(src, spec, embodiment_tag=overlay.get("embodiment_tag"))
    else:
        spec = {"lerobot": _lerobot, "openpi": _openpi, "custom": _custom}[family](src, spec)
    if overlay:
        spec = _apply_overlay(spec, overlay)
    spec.unverified = sorted(set(spec.unverified))
    return spec

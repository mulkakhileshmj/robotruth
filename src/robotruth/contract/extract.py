"""Adapters that build an ExecSpec from a checkpoint directory.

Adapters never import the policy framework. They read files: config.json,
train_config.json, safetensors headers, norm-stats json. Anything they cannot read becomes
`unknown` and is listed in `spec.unverified`, so the checker can fail closed.

Families:
- lerobot: a `pretrained_model/` directory with config.json, model.safetensors and
  (optionally) train_config.json. Normalization statistics may live in
  `policy_preprocessor*.safetensors`/json (LeRobot 0.4+), in `stats.json`, or inside the
  weights as `normalize_*` buffers (older versions).
- openpi: a checkpoint directory containing `assets/<repo_id>/norm_stats.json` and
  orbax `params/`; config lives in code, so only what is present on disk is recorded.
- gr00t: a HF-style directory with config.json, model.safetensors and an experiment
  config describing modality and action horizon.
- custom: hashes every file and reads a user-supplied `robotruth.spec.json` overlay.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Optional

from robotruth.contract.spec import (
    ActionSemantics,
    CameraSpec,
    Embodiment,
    ExecSpec,
    Normalization,
    ObservationSpec,
    PolicyIdentity,
    RuntimeSpec,
    sha256_file,
    sha256_files,
    sha256_obj,
)

WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".msgpack", ".npz"}


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safetensors_header(path: Path) -> Optional[dict]:
    """Read only the JSON header of a safetensors file (no tensor data, no torch)."""
    try:
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n).decode("utf-8"))
    except Exception:
        return None


def _weights(dir_: Path) -> list[Path]:
    files: list[Path] = []
    for p in dir_.rglob("*"):
        if p.is_file() and p.suffix.lower() in WEIGHT_SUFFIXES and "preprocessor" not in p.name and "postprocessor" not in p.name:
            files.append(p)
    return files


def _norm_type_from_mapping(mapping: Optional[dict], key_hint: str) -> str:
    if not mapping:
        return "unknown"
    for k, v in mapping.items():
        if key_hint.lower() in k.lower():
            v = str(v).upper()
            if "MEAN" in v:
                return "mean_std"
            if "MIN" in v:
                return "min_max"
            if "QUANT" in v:
                return "quantile"
            if "IDENT" in v:
                return "identity"
    return "unknown"


def _find_stats_hash(dir_: Path, weights_header: Optional[dict]) -> tuple[Optional[str], Optional[str]]:
    """Locate normalization statistics and return (sha256, source description)."""
    candidates = []
    for name in ("stats.json", "norm_stats.json", "dataset_stats.json"):
        for p in dir_.rglob(name):
            candidates.append(p)
    for p in dir_.rglob("*preprocessor*"):
        if p.is_file():
            candidates.append(p)
    for p in dir_.rglob("*postprocessor*"):
        if p.is_file():
            candidates.append(p)
    if candidates:
        return sha256_files(candidates), ",".join(sorted(c.name for c in candidates))
    if weights_header:
        norm_keys = sorted(k for k in weights_header if k.startswith("normalize_") or k.startswith("unnormalize_"))
        if norm_keys:
            # Hash the tensor metadata (names, shapes, offsets); content is inside the weights hash.
            return sha256_obj({k: weights_header[k] for k in norm_keys}), "weights:normalize_* buffers"
    return None, None


def _lerobot(dir_: Path, spec: ExecSpec) -> ExecSpec:
    cfg = _read_json(dir_ / "config.json") or {}
    train_cfg = _read_json(dir_ / "train_config.json") or {}
    weights = _weights(dir_)
    header = _safetensors_header(weights[0]) if weights and weights[0].suffix == ".safetensors" else None

    spec.policy.family = "lerobot"
    spec.policy.name = str(cfg.get("type") or train_cfg.get("policy", {}).get("type") or dir_.name)
    if weights:
        spec.policy.weights_files = [str(p.relative_to(dir_)) for p in weights]
        spec.policy.weights_sha256 = sha256_files(weights)
        spec.policy.weights_bytes = sum(p.stat().st_size for p in weights)
    else:
        spec.unverified.append("policy.weights_sha256")
    if (dir_ / "config.json").exists():
        spec.policy.config_sha256 = sha256_file(dir_ / "config.json")

    out_feats = cfg.get("output_features") or {}
    in_feats = cfg.get("input_features") or {}
    action_feat = out_feats.get("action") or {}
    shape = action_feat.get("shape")
    spec.action.dim = int(shape[0]) if shape else None
    spec.action.chunk_size = cfg.get("chunk_size") or cfg.get("horizon") or cfg.get("action_horizon")
    spec.action.n_action_steps = cfg.get("n_action_steps")
    spec.action.control_hz = (train_cfg.get("dataset") or {}).get("fps") or cfg.get("fps")
    nm = cfg.get("normalization_mapping") or {}
    spec.action.normalization.type = _norm_type_from_mapping(nm, "ACTION")
    spec.observation.normalization.type = _norm_type_from_mapping(nm, "STATE")
    stats_hash, stats_src = _find_stats_hash(dir_, header)
    spec.action.normalization.stats_sha256 = stats_hash
    spec.action.normalization.stats_source = stats_src
    spec.observation.normalization.stats_sha256 = stats_hash
    spec.observation.normalization.stats_source = stats_src
    spec.action.normalization.action_dim = spec.action.dim
    # LeRobot policies are absolute joint targets unless the config says otherwise.
    if cfg.get("use_delta_joint_actions") is True or cfg.get("relative_actions") is True:
        spec.action.mode = "delta"
    elif cfg:
        spec.action.mode = "absolute"
    spec.action.space = "joint_position" if cfg else "unknown"

    for key, feat in in_feats.items():
        if str(feat.get("type", "")).upper() == "VISUAL" or "image" in key:
            sh = feat.get("shape") or [None, None, None]
            spec.observation.cameras.append(CameraSpec(name=key, channels=sh[0], height=sh[1], width=sh[2]))
        elif "state" in key:
            spec.observation.state_keys.append(key)
            sh = feat.get("shape")
            spec.observation.state_dim = int(sh[0]) if sh else None
    resize = cfg.get("resize_imgs_with_padding") or cfg.get("image_resize") or cfg.get("crop_shape")
    spec.observation.image_resize = str(resize) if resize else None
    spec.observation.preprocessing_sha256 = sha256_obj({
        "resize": resize, "crop_is_random": cfg.get("crop_is_random"),
        "normalization_mapping": nm, "input_features": in_feats,
    }) if cfg else None
    spec.runtime.framework = "lerobot"
    spec.runtime.dtype = cfg.get("dtype") or ("bfloat16" if cfg.get("use_amp") else None)
    spec.embodiment.robot = str((train_cfg.get("env") or {}).get("type") or (train_cfg.get("dataset") or {}).get("repo_id") or "unknown")

    for f, v in (
        ("action.dim", spec.action.dim), ("action.chunk_size", spec.action.chunk_size),
        ("action.n_action_steps", spec.action.n_action_steps), ("action.control_hz", spec.action.control_hz),
        ("action.normalization.stats_sha256", stats_hash), ("observation.state_dim", spec.observation.state_dim),
    ):
        if v is None:
            spec.unverified.append(f)
    spec.unverified.append("action.gripper_convention")
    return spec


def _openpi(dir_: Path, spec: ExecSpec) -> ExecSpec:
    spec.policy.family = "openpi"
    spec.policy.name = dir_.name
    weights = [p for p in dir_.rglob("*") if p.is_file() and ("params" in p.parts or p.suffix in WEIGHT_SUFFIXES)]
    if weights:
        spec.policy.weights_files = [str(p.relative_to(dir_)) for p in weights[:50]]
        spec.policy.weights_sha256 = sha256_files(weights)
        spec.policy.weights_bytes = sum(p.stat().st_size for p in weights)
    else:
        spec.unverified.append("policy.weights_sha256")
    stats = list(dir_.rglob("norm_stats.json"))
    if stats:
        spec.action.normalization.stats_sha256 = sha256_files(stats)
        spec.action.normalization.stats_source = ",".join(str(s.relative_to(dir_)) for s in stats)
        st = _read_json(stats[0]) or {}
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
    spec.unverified += ["action.chunk_size", "action.control_hz", "action.mode", "action.space", "observation.preprocessing_sha256", "observation.state_dim", "action.gripper_convention"]
    return spec


def _gr00t(dir_: Path, spec: ExecSpec) -> ExecSpec:
    spec.policy.family = "gr00t"
    cfg = _read_json(dir_ / "config.json") or {}
    exp = _read_json(dir_ / "experiment_cfg" / "metadata.json") or {}
    spec.policy.name = str(cfg.get("model_type") or dir_.name)
    weights = _weights(dir_)
    if weights:
        spec.policy.weights_files = [str(p.relative_to(dir_)) for p in weights]
        spec.policy.weights_sha256 = sha256_files(weights)
        spec.policy.weights_bytes = sum(p.stat().st_size for p in weights)
    else:
        spec.unverified.append("policy.weights_sha256")
    if (dir_ / "config.json").exists():
        spec.policy.config_sha256 = sha256_file(dir_ / "config.json")
    spec.action.chunk_size = cfg.get("action_horizon")
    spec.action.dim = cfg.get("action_dim")
    if exp:
        spec.action.normalization.stats_sha256 = sha256_obj(exp)
        spec.action.normalization.stats_source = "experiment_cfg/metadata.json"
        spec.observation.normalization.stats_sha256 = spec.action.normalization.stats_sha256
        spec.embodiment.robot = ",".join(sorted(exp.keys())) if isinstance(exp, dict) else "unknown"
    else:
        spec.unverified.append("action.normalization.stats_sha256")
    spec.runtime.framework = "gr00t"
    spec.unverified += ["action.control_hz", "action.mode", "action.space", "observation.preprocessing_sha256", "observation.state_dim", "action.gripper_convention", "action.n_action_steps"]
    return spec


def _custom(dir_: Path, spec: ExecSpec) -> ExecSpec:
    spec.policy.family = "custom"
    spec.policy.name = dir_.name
    weights = _weights(dir_)
    if weights:
        spec.policy.weights_files = [str(p.relative_to(dir_)) for p in weights]
        spec.policy.weights_sha256 = sha256_files(weights)
        spec.policy.weights_bytes = sum(p.stat().st_size for p in weights)
    else:
        spec.unverified.append("policy.weights_sha256")
    spec.unverified += ["action.normalization.stats_sha256", "action.chunk_size", "action.control_hz", "action.mode", "action.space", "action.dim", "observation.preprocessing_sha256", "observation.state_dim", "action.gripper_convention", "action.n_action_steps"]
    return spec


def _apply_overlay(spec: ExecSpec, overlay: dict[str, Any]) -> ExecSpec:
    """Merge a user-supplied partial manifest (dotted or nested) and drop resolved unverified entries."""
    data = spec.model_dump()

    def set_dotted(d: dict, dotted: str, value: Any):
        parts = dotted.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value

    def flatten(prefix: str, node: Any):
        if isinstance(node, dict) and not (prefix.endswith("cameras")):
            for k, v in node.items():
                flatten(f"{prefix}.{k}" if prefix else k, v)
        else:
            set_dotted(data, prefix, node)
            data_unverified = data.get("unverified", [])
            if prefix in data_unverified:
                data_unverified.remove(prefix)

    for k, v in overlay.items():
        flatten(k, v)
    if "observation" in overlay and isinstance(overlay["observation"], dict) and "cameras" in overlay["observation"]:
        data["observation"]["cameras"] = overlay["observation"]["cameras"]
    return ExecSpec.model_validate(data)


def detect_family(dir_: Path) -> str:
    if (dir_ / "config.json").exists():
        cfg = _read_json(dir_ / "config.json") or {}
        if "action_horizon" in cfg or "gr00t" in str(cfg.get("model_type", "")).lower() or (dir_ / "experiment_cfg").exists():
            return "gr00t"
        if "input_features" in cfg or "output_features" in cfg or "normalization_mapping" in cfg:
            return "lerobot"
    if list(dir_.rglob("norm_stats.json")) or (dir_ / "params").exists():
        return "openpi"
    return "custom"


def extract_spec(path: Path | str, family: Optional[str] = None, overlay: Optional[dict[str, Any]] = None,
                 role: str = "unspecified") -> ExecSpec:
    dir_ = Path(path)
    if not dir_.exists():
        raise FileNotFoundError(dir_)
    if dir_.is_file():
        dir_ = dir_.parent
    family = family or detect_family(dir_)
    spec = ExecSpec(role=role)  # type: ignore[arg-type]
    spec.policy.source_path = str(dir_)
    fn = {"lerobot": _lerobot, "openpi": _openpi, "gr00t": _gr00t, "custom": _custom}[family]
    spec = fn(dir_, spec)
    overlay_file = dir_ / "robotruth.spec.json"
    if overlay_file.exists():
        spec = _apply_overlay(spec, _read_json(overlay_file) or {})
    if overlay:
        spec = _apply_overlay(spec, overlay)
    spec.unverified = sorted(set(spec.unverified))
    return spec

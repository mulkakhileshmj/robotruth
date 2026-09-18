"""Video judges: the VLM half of the hybrid outcome judge.

A backend looks at a handful of frames plus the instruction and returns a verdict with a
success probability, a progress estimate and a failure class guess. Two backends ship:

- `MockBackend`: deterministic, configurable, for tests and for dry runs without a key.
- `AnthropicBackend`: sends subsampled frames to Claude with a strict JSON prompt. The
  prompt warns the model about the documented bias toward calling success (FailBench,
  arXiv 2609.03611) and asks for evidence of the final state rather than motion.

The `anthropic` package is imported lazily inside `AnthropicBackend` so the rest of
robotruth never depends on it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import struct
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from robotruth.schema.episode import FailureClass

DEFAULT_MODEL = "claude-opus-5"
FAILURE_CLASS_VALUES = [c.value for c in FailureClass]


@dataclass
class VLMVerdict:
    """What a video judge says about one episode."""

    success_prob: float
    progress: float
    failure_class: Optional[str] = None
    rationale: str = ""
    raw: dict = field(default_factory=dict)
    parse_error: bool = False
    model: str = "unknown"

    def __post_init__(self):
        self.success_prob = float(np.clip(self.success_prob, 0.0, 1.0))
        self.progress = float(np.clip(self.progress, 0.0, 1.0))
        if self.failure_class is not None:
            fc = str(self.failure_class).strip().lower()
            self.failure_class = fc if fc in FAILURE_CLASS_VALUES else None

    def to_dict(self) -> dict:
        return asdict(self)


@runtime_checkable
class VLMBackend(Protocol):
    """Anything that can judge an episode from frames and an instruction."""

    name: str

    def judge(self, frames: Optional[Sequence[Any]], instruction: str, task: str) -> VLMVerdict: ...


def _stable_unit(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from strings. Same inputs, same output."""
    h = hashlib.sha1("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


class MockBackend:
    """Deterministic stand-in for a VLM.

    Priority of behaviour: `fn(frames, instruction, task)` if given, else the entry of
    `verdicts` keyed by instruction, else the constant `success_prob` plus an optional
    deterministic `jitter` derived from the instruction and task.
    """

    name = "mock"

    def __init__(self, success_prob: float = 0.5, progress: Optional[float] = None,
                 failure_class: Optional[str] = None, jitter: float = 0.0,
                 verdicts: Optional[dict[str, VLMVerdict]] = None,
                 fn: Optional[Callable[[Optional[Sequence[Any]], str, str], float]] = None):
        self.success_prob = success_prob
        self.progress = progress
        self.failure_class = failure_class
        self.jitter = jitter
        self.verdicts = verdicts or {}
        self.fn = fn
        self.calls = 0

    def judge(self, frames: Optional[Sequence[Any]], instruction: str, task: str) -> VLMVerdict:
        self.calls += 1
        if self.fn is not None:
            p = float(self.fn(frames, instruction, task))
        elif instruction in self.verdicts:
            v = self.verdicts[instruction]
            return VLMVerdict(v.success_prob, v.progress, v.failure_class, v.rationale, dict(v.raw), v.parse_error, "mock")
        else:
            p = self.success_prob
            if self.jitter > 0:
                p += self.jitter * (2 * _stable_unit(instruction, task) - 1)
        p = float(np.clip(p, 0.0, 1.0))
        progress = self.progress if self.progress is not None else p
        fc = self.failure_class if p < 0.5 else None
        return VLMVerdict(p, progress, fc, "mock verdict", {"n_frames": 0 if frames is None else len(frames)}, False, "mock")


# ----------------------------------------------------------------------------- frames

def subsample_indices(n: int, k: int) -> list[int]:
    """Indices of the first, the last and k-2 evenly spaced frames in between. Sorted, unique."""
    if n <= 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [n - 1]
    idx = np.linspace(0, n - 1, num=k)
    return sorted(set(int(round(i)) for i in idx))


def encode_png(arr: np.ndarray) -> bytes:
    """Minimal PNG encoder for uint8 HxW (grey), HxWx3 (RGB) or HxWx4 (RGBA) arrays. No PIL."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8) if a.max() > 1.0 else (a * 255).astype(np.uint8)
    if a.ndim == 2:
        color_type, channels = 0, 1
        a = a[:, :, None]
    elif a.ndim == 3 and a.shape[2] == 3:
        color_type, channels = 2, 3
    elif a.ndim == 3 and a.shape[2] == 4:
        color_type, channels = 6, 4
    elif a.ndim == 3 and a.shape[2] == 1:
        color_type, channels = 0, 1
    else:
        raise ValueError(f"cannot encode array of shape {a.shape} as PNG")
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


_MEDIA_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}


def frame_to_image_block(frame: Any) -> dict:
    """Turn one frame (array, path or bytes) into an Anthropic image content block."""
    if isinstance(frame, (str, Path)):
        p = Path(frame)
        media = _MEDIA_BY_EXT.get(p.suffix.lower())
        data = p.read_bytes()
        if media is None:
            media = _sniff_media(data)
    elif isinstance(frame, (bytes, bytearray)):
        data = bytes(frame)
        media = _sniff_media(data)
    else:
        data = encode_png(np.asarray(frame))
        media = "image/png"
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": base64.standard_b64encode(data).decode("ascii")}}


def _sniff_media(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"


# ----------------------------------------------------------------------------- prompt

SYSTEM_PROMPT = """You are a strict outcome judge for robot manipulation episodes.

You will see a small number of frames sampled from one episode, in time order, first frame to last frame, together with the instruction the robot was given. Decide whether the task was completed.

Known failure mode of judges like you, measured on 2,197 real attempts: a systematic bias toward calling success. Judges see the arm move toward the object and conclude the task worked. Do not do this. Motion is not completion. Reaching is not grasping. Lifting is not placing. A gripper closed on nothing looks like a grasp from many angles.

Rules:
1. Judge the FINAL STATE, mainly from the last frames. Ask: is the world in the state the instruction requires? Where is the object now? Is it inside, on, or attached to what it should be? Is the gripper empty at the end when it should be?
2. If the final frames do not show clear evidence of the required end state, the episode is not a success. Absence of evidence counts against success, not for it.
3. Contact-rich steps (insertion, stacking, closing a drawer fully, screwing) fail silently. Look for misalignment, gaps, tilt, objects resting on edges, half-open drawers, caps not seated.
4. Your success_prob must be calibrated: 0.9 means you would be wrong about one time in ten on episodes that look like this one. Use the middle of the range when the frames are ambiguous, occluded or too few.
5. progress is how far along the task got in [0, 1], independent of whether it finished.
6. failure_class is one of the listed values or null when the episode succeeded or you cannot tell.

Return ONLY a JSON object, no prose before or after, with exactly these keys:
{"success_prob": <number 0..1>, "progress": <number 0..1>, "failure_class": <string or null>, "final_state_evidence": "<one sentence on what the last frames show>", "rationale": "<one or two short sentences>"}"""


def build_user_text(instruction: str, task: str, n_frames: int) -> str:
    classes = ", ".join(FAILURE_CLASS_VALUES)
    return (
        f"Task id: {task or 'unknown'}\n"
        f"Instruction given to the robot: {instruction or '(none)'}\n"
        f"Frames: {n_frames}, in time order, the last one is the final state of the episode.\n"
        f"Allowed failure_class values: {classes}.\n"
        "Judge the final state, not the motion. Respond with the JSON object only."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str, model: str = "unknown") -> VLMVerdict:
    """Parse the JSON verdict out of a model reply. On any problem, return an abstain-shaped
    verdict: success_prob 0.5, progress 0.5, parse_error True, the raw text kept in `raw`."""
    fallback = VLMVerdict(0.5, 0.5, None, "could not parse the judge reply", {"text": text}, True, model)
    if not text:
        return fallback
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        m = _JSON_RE.search(candidate)
        if not m:
            return fallback
        candidate = m.group(0)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(obj, dict) or "success_prob" not in obj:
        return fallback
    try:
        p = float(obj["success_prob"])
        progress = float(obj.get("progress", p))
    except (TypeError, ValueError):
        return fallback
    if not (np.isfinite(p) and np.isfinite(progress)):
        return fallback
    rationale = " ".join(str(obj.get(k, "")).strip() for k in ("final_state_evidence", "rationale") if obj.get(k))
    return VLMVerdict(p, progress, obj.get("failure_class"), rationale[:500], dict(obj), False, model)


class AnthropicBackend:
    """Video judge backed by Claude.

    Frames are subsampled to `n_frames` (first, last and evenly spaced in between) and sent
    as images with a strict JSON prompt. Refusals, API errors and unparseable replies all
    come back as a verdict with success_prob 0.5 and `parse_error=True`, so a batch never
    dies on one episode and the calibrator can treat those as abstain.

    The API key is read from ANTHROPIC_API_KEY, or from a .env at the project root via
    robotruth._env.load_env. Pass `client` to inject a client (tests use a fake).
    Server-side refusal fallbacks are enabled by default (`fallbacks=True`); they route a
    declined request to another model inside the same call.
    """

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, n_frames: int = 8, max_tokens: int = 2048,
                 fallbacks: bool = True, client: Any = None, effort: Optional[str] = None):
        self.model = model
        self.n_frames = max(1, int(n_frames))
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self.effort = effort
        self._client = client

    def _client_or_create(self):
        if self._client is None:
            try:
                from robotruth._env import load_env
                load_env(Path(__file__).resolve().parents[3] / ".env")
            except Exception:
                pass
            if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
                # The SDK also resolves `ant auth login` profiles, so only warn, do not fail here.
                pass
            import anthropic  # lazy: not a core dependency

            self._client = anthropic.Anthropic()
        return self._client

    def build_messages(self, frames: Optional[Sequence[Any]], instruction: str, task: str) -> list[dict]:
        frames = list(frames or [])
        idx = subsample_indices(len(frames), self.n_frames)
        content: list[dict] = []
        for i in idx:
            content.append({"type": "text", "text": f"frame {i + 1} of {len(frames)}"})
            content.append(frame_to_image_block(frames[i]))
        content.append({"type": "text", "text": build_user_text(instruction, task, len(idx))})
        return [{"role": "user", "content": content}]

    def judge(self, frames: Optional[Sequence[Any]], instruction: str, task: str) -> VLMVerdict:
        if not frames:
            return VLMVerdict(0.5, 0.5, None, "no frames supplied", {"error": "no_frames"}, True, self.model)
        try:
            client = self._client_or_create()
            messages = self.build_messages(frames, instruction, task)
            kwargs: dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM_PROMPT, messages=messages)
            if self.effort:
                kwargs["output_config"] = {"effort": self.effort}
            if self.fallbacks:
                response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                response = client.messages.create(**kwargs)
        except Exception as e:  # network, auth, bad request: report, do not crash the batch
            return VLMVerdict(0.5, 0.5, None, f"api error: {type(e).__name__}", {"error": f"{type(e).__name__}: {e}"[:500]}, True, self.model)

        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            raw = {"stop_reason": "refusal", "category": getattr(details, "category", None), "explanation": getattr(details, "explanation", None)}
            return VLMVerdict(0.5, 0.5, None, "judge refused", raw, True, getattr(response, "model", self.model))
        text = "".join(getattr(b, "text", "") for b in getattr(response, "content", []) if getattr(b, "type", "") == "text")
        verdict = parse_verdict(text, getattr(response, "model", self.model))
        usage = getattr(response, "usage", None)
        if usage is not None:
            verdict.raw["usage"] = {"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None)}
        verdict.raw["stop_reason"] = stop
        return verdict

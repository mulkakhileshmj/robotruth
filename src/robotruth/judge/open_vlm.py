"""Open-weight VLM backend for the judge: no API key, runs on your own GPU.

Uses a Qwen-VL class instruction model through transformers. The same strict-JSON prompt,
frame subsampling and parser as the Anthropic backend, so the two are interchangeable and
comparable. `transformers` and `torch` are imported lazily; nothing else in robotruth
depends on them.

Model choice: Qwen/Qwen2.5-VL-7B-Instruct by default (fits a 24 GB GPU in bf16). Any model
loadable by AutoModelForImageTextToText with an AutoProcessor chat template that accepts
image content parts should work; pass `model_id` to swap.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from robotruth.judge.vlm import (
    SYSTEM_PROMPT,
    VLMVerdict,
    build_user_text,
    parse_verdict,
    subsample_indices,
)

DEFAULT_OPEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def _to_pil(frame: Any):
    """Frame (ndarray HxWx3, path, or bytes) to a PIL image."""
    from PIL import Image
    if isinstance(frame, (str, Path)):
        return Image.open(frame).convert("RGB")
    if isinstance(frame, (bytes, bytearray)):
        return Image.open(io.BytesIO(frame)).convert("RGB")
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 + 1e-6 else 1.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


class OpenVLMBackend:
    """Local open-weight video judge. Free per call; you pay only for the GPU it runs on."""

    def __init__(self, model_id: str = DEFAULT_OPEN_MODEL, n_frames: int = 8,
                 max_new_tokens: int = 512, device: str = "auto", dtype: str = "bfloat16",
                 max_image_side: int = 640):
        self.model_id = model_id
        self.name = f"open:{model_id}"
        self.n_frames = n_frames
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.dtype = dtype
        self.max_image_side = max_image_side
        self._model = None
        self._processor = None

    # ---- lazy loading -------------------------------------------------------------
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=self.device)
        self._model.eval()

    # ---- inference ----------------------------------------------------------------
    def _prepare_images(self, frames: Sequence[Any]) -> list:
        idx = subsample_indices(len(frames), self.n_frames)
        images = []
        for i in idx:
            img = _to_pil(frames[i])
            if max(img.size) > self.max_image_side:
                scale = self.max_image_side / max(img.size)
                img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
            images.append(img)
        return images

    def judge(self, frames: Optional[Sequence[Any]], instruction: str, task: str) -> VLMVerdict:
        if not frames:
            return VLMVerdict(0.5, 0.5, None, "no frames provided", {"error": "no_frames"}, True, self.name)
        try:
            self._load()
            import torch
            images = self._prepare_images(frames)
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": build_user_text(instruction, task, len(images))})
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content},
            ]
            inputs = self._processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            text = self._processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            return parse_verdict(text, model=self.name)
        except Exception as e:  # noqa: BLE001
            return VLMVerdict(0.5, 0.5, None, f"backend error: {type(e).__name__}: {e}",
                              {"error": str(e)[:500]}, True, self.name)

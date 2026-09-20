"""ACT transfer-cube policy adapter (lerobot/act_aloha_sim_transfer_cube_human).

Unmodified public weights. lerobot 0.6.1 silently drops this checkpoint's normalization
buffers on load, which turns an ~83% policy into 0%; the robotruth live-loop validation
found this in the wild. The adapter reattaches the statistics from the checkpoint's own
safetensors, exactly as ops/live_guard_test.py does.

Controlled variants (never touching the weights) provide the demo policy B:
  variant={"state_bias": x}  adds a constant proprioception bias (mild degradation)
  variant={"drop_norm": true}  reproduces the normalization bug (catastrophic)

Runs on the GPU box only.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from policyci.policy_iface import PolicyContract

MODEL_ID = "lerobot/act_aloha_sim_transfer_cube_human"


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ACTAlohaPolicy:
    def __init__(self, name: str, device: str = "cuda", variant: dict | None = None):
        import torch
        from huggingface_hub import hf_hub_download
        from lerobot.policies.act.modeling_act import ACTPolicy
        from safetensors.torch import load_file

        self._torch = torch
        self._device = device
        variant = dict(variant or {})

        self._policy = ACTPolicy.from_pretrained(MODEL_ID)
        self._policy.to(device)
        self._policy.eval()

        weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
        sd = load_file(weights_path)
        if variant.get("drop_norm"):
            self._stats = None  # run the policy the way the buggy loader does
        else:
            self._stats = {
                "img_mean": sd["normalize_inputs.buffer_observation_images_top.mean"].to(device),
                "img_std": sd["normalize_inputs.buffer_observation_images_top.std"].to(device),
                "state_mean": sd["normalize_inputs.buffer_observation_state.mean"].to(device),
                "state_std": sd["normalize_inputs.buffer_observation_state.std"].to(device),
                "act_mean": sd["unnormalize_outputs.buffer_action.mean"].to(device),
                "act_std": sd["unnormalize_outputs.buffer_action.std"].to(device),
            }
        self._state_bias = float(variant.get("state_bias", 0.0))

        self.contract = PolicyContract(
            name=name,
            model_id=MODEL_ID,
            action_dim=14,
            action_semantics="joint_position",
            control_hz=50.0,
            obs_keys=("observation.images.top", "observation.state"),
            weights_sha256=_sha256_file(weights_path),
            variant=variant,
        )

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._torch.manual_seed(seed)
        self._policy.reset()

    def act(self, obs: Any) -> np.ndarray:
        torch = self._torch
        img = obs["pixels"]["top"].astype(np.float32) / 255.0
        state = obs["agent_pos"].astype(np.float32) + self._state_bias
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
        state_t = torch.from_numpy(state).unsqueeze(0).to(self._device)
        if self._stats is not None:
            img_t = (img_t - self._stats["img_mean"]) / (self._stats["img_std"] + 1e-8)
            state_t = (state_t - self._stats["state_mean"]) / (self._stats["state_std"] + 1e-8)
            batch = {"observation.images.top": img_t, "observation.state": state_t}
            with torch.no_grad():
                action = self._policy.select_action(batch)
            action = action * (self._stats["act_std"] + 1e-8) + self._stats["act_mean"]
        else:
            batch = {"observation.images.top": img_t, "observation.state": state_t}
            with torch.no_grad():
                action = self._policy.select_action(batch)
        return action.squeeze(0).cpu().numpy()

"""Policy and environment combinations for the live guard experiments. GPU box only.

The 0.1.4 guard numbers come from one policy on one task (ACT on aloha transfer cube),
which is the single biggest generalisation gap in the published envelope. This module makes
the combination a parameter so the same experiment can run on other policies and tasks.

Every combo supplies four things: a gym environment, a loaded policy, the normalization
statistics the checkpoint shipped, and an adapter that turns an observation into the batch
that policy expects. `smoke_test` runs the policy for a few episodes and reports its
success rate, because calibrating a guard on a policy that does not work produces a
meaningless nominal distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class Combo:
    name: str
    repo: str
    env_id: str
    policy_class: str          # "act" or "diffusion"
    image_key: str             # policy input key for the camera
    obs_image_path: tuple      # where the frame lives in the observation dict
    max_steps: int = 400
    fps: float = 50.0
    expected_success: float = 0.5   # from the model card; the smoke test compares against it
    notes: str = ""
    extra: dict = field(default_factory=dict)


COMBOS: dict[str, Combo] = {
    "act_aloha_transfer": Combo(
        name="act_aloha_transfer",
        repo="lerobot/act_aloha_sim_transfer_cube_human",
        env_id="gym_aloha/AlohaTransferCube-v0",
        policy_class="act",
        image_key="observation.images.top",
        obs_image_path=("pixels", "top"),
        max_steps=400, fps=50.0, expected_success=0.8,
        notes="the 0.1.3 and 0.1.4 reference combination",
    ),
    "act_aloha_insertion": Combo(
        name="act_aloha_insertion",
        repo="lerobot/act_aloha_sim_insertion_human",
        env_id="gym_aloha/AlohaInsertion-v0",
        policy_class="act",
        image_key="observation.images.top",
        obs_image_path=("pixels", "top"),
        max_steps=400, fps=50.0, expected_success=0.3,
        notes="second task, same policy architecture and checkpoint family",
    ),
    "diffusion_pusht": Combo(
        name="diffusion_pusht",
        repo="lerobot/diffusion_pusht",
        env_id="gym_pusht/PushT-v0",
        policy_class="diffusion",
        image_key="observation.image",
        obs_image_path=("pixels",),
        max_steps=300, fps=10.0, expected_success=0.5,
        notes="different architecture, different task, different control rate",
    ),
}


def make_env(combo: Combo):
    import gymnasium as gym
    if combo.env_id.startswith("gym_aloha"):
        import gym_aloha  # noqa: F401
    elif combo.env_id.startswith("gym_pusht"):
        import gym_pusht  # noqa: F401
    return gym.make(combo.env_id, obs_type="pixels_agent_pos", max_episode_steps=combo.max_steps)


def load_policy(combo: Combo, device: str):
    """Load the checkpoint and reattach the normalization statistics it shipped.

    lerobot drops some older checkpoints' normalize_* buffers on load with only a warning,
    which silently runs the policy unnormalized. That is the contract failure class this
    project exists to catch, so the statistics are read straight out of the checkpoint's
    safetensors and applied here instead of trusting the loader.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    if combo.policy_class == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as Cls
    elif combo.policy_class == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as Cls
    else:
        raise ValueError(f"unknown policy class {combo.policy_class}")

    policy = Cls.from_pretrained(combo.repo)
    policy.to(device)
    policy.eval()

    sd = load_file(hf_hub_download(combo.repo, "model.safetensors"))
    stats: dict[str, Any] = {}
    for key, tensor in sd.items():
        if ".buffer_" not in key:
            continue
        group, rest = key.split(".buffer_", 1)
        feature, _, stat = rest.rpartition(".")
        stats[f"{group}|{feature}|{stat}"] = tensor.to(device)

    def pick(feature: str, stat: str):
        for group in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
            v = stats.get(f"{group}|{feature}|{stat}")
            if v is not None:
                return v
        return None

    img_feat = combo.image_key.replace(".", "_")
    resolved = {
        "img_mean": pick(img_feat, "mean"), "img_std": pick(img_feat, "std"),
        "state_mean": pick("observation_state", "mean"), "state_std": pick("observation_state", "std"),
        "act_mean": pick("action", "mean"), "act_std": pick("action", "std"),
    }
    missing = [k for k, v in resolved.items() if v is None]
    if missing:
        raise RuntimeError(f"{combo.repo}: normalization statistics missing for {missing}; "
                           "this checkpoint cannot be run faithfully, which is itself a contract failure")
    return policy, resolved


def obs_to_batch(combo: Combo, obs, device, torch):
    frame = obs
    for k in combo.obs_image_path:
        frame = frame[k]
    img = np.asarray(frame, dtype=np.float32) / 255.0
    state = np.asarray(obs["agent_pos"], dtype=np.float32)
    return {
        combo.image_key: torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device),
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }


def normalize_batch(combo: Combo, batch: dict, stats: dict):
    out = dict(batch)
    out[combo.image_key] = (batch[combo.image_key] - stats["img_mean"]) / (stats["img_std"] + 1e-8)
    out["observation.state"] = (batch["observation.state"] - stats["state_mean"]) / (stats["state_std"] + 1e-8)
    return out


def is_success(combo: Combo, max_reward: float, info: dict) -> bool:
    """Task success from the simulator's own signal."""
    if combo.env_id.startswith("gym_aloha"):
        return bool(max_reward >= 4.0)
    if combo.env_id.startswith("gym_pusht"):
        return bool(info.get("is_success", False) or max_reward >= 0.95)
    return bool(max_reward > 0)


def run_episode(combo: Combo, env, policy_and_stats, device, torch,
                fault: Optional[str] = None, rng=None, chunk_k: int = 10,
                onset_step: int = 150, ramp_steps: int = 0,
                offset: float = 0.15, noise_sigma: float = 0.2):
    """One live episode with an optional executed-action fault.

    ramp_steps > 0 ramps the fault in linearly over that many steps instead of applying it
    at full strength immediately, which is what a degrading robot actually looks like. A
    freeze ramps by blending the held action with the fresh one.
    """
    policy, stats = policy_and_stats
    rng = rng if rng is not None else np.random.default_rng(0)
    obs, _ = env.reset()
    policy.reset()
    feats, acts = [], []
    max_reward, steps, held = 0.0, 0, None
    last_info: dict = {}
    for t in range(combo.max_steps):
        batch = normalize_batch(combo, obs_to_batch(combo, obs, device, torch), stats)
        with torch.no_grad():
            action = policy.select_action(batch)
        action = action * (stats["act_std"] + 1e-8) + stats["act_mean"]
        a = action.squeeze(0).cpu().numpy().astype(np.float32)
        strength = 0.0
        if fault and t >= onset_step:
            strength = 1.0 if ramp_steps <= 0 else min(1.0, (t - onset_step + 1) / ramp_steps)
        if strength > 0.0:
            if fault == "freeze":
                if held is not None:
                    a = (1.0 - strength) * a + strength * held
            elif fault == "offset":
                a = a + strength * offset
            elif fault == "noise":
                a = a + strength * rng.normal(0.0, noise_sigma, size=a.shape).astype(np.float32)
            else:
                raise ValueError(f"unknown fault {fault}")
            a = a.astype(np.float32)
        # `held` is the frozen reference for a stall: it tracks the executed action until the
        # freeze engages, then stops so the arm holds the pose it had when the fault began.
        if not (fault == "freeze" and strength > 0.0):
            held = a.copy()
        state = np.asarray(obs["agent_pos"], dtype=np.float32)
        feats.append(np.concatenate([state, a]))
        acts.append(a)
        obs, reward, terminated, truncated, info = env.step(a)
        last_info = info or {}
        max_reward = max(max_reward, float(reward))
        steps = t + 1
        if terminated or truncated:
            break
    arr = np.asarray(acts)
    chunks = np.stack([arr[max(0, i - chunk_k + 1): i + 1] if i + 1 >= chunk_k else
                       np.pad(arr[: i + 1], ((chunk_k - i - 1, 0), (0, 0)), mode="edge")
                       for i in range(len(arr))])
    return {"features": np.asarray(feats), "actions": chunks, "timestamps": np.arange(steps) / combo.fps,
            "success": is_success(combo, max_reward, last_info), "steps": steps,
            "duration_s": steps / combo.fps, "fault": fault or "none", "ramp_steps": ramp_steps}


def smoke_test(combo: Combo, env, policy_and_stats, device, torch, n: int = 10) -> tuple[float, bool]:
    """Run the policy unfaulted and report its success rate before anything is calibrated."""
    ok = 0
    for _ in range(n):
        ok += bool(run_episode(combo, env, policy_and_stats, device, torch)["success"])
    rate = ok / n
    usable = rate >= max(0.2, 0.4 * combo.expected_success)
    print(f"smoke test {combo.name}: {ok}/{n} successes (model card suggests about "
          f"{combo.expected_success:.0%}), usable={usable}", flush=True)
    return rate, usable

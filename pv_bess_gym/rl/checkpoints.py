"""Load RL checkpoints as plain ``predict(flat_obs) -> action`` functions.

Two families of checkpoints exist: Stable-Baselines3 (PPO, SAC) and OmniSafe
(PPO-Lagrangian). Each loader imports its library lazily, so this module can
be imported in either Python environment.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

Predict = Callable[[np.ndarray], np.ndarray]


def list_checkpoints(run_dir: str | Path) -> list[tuple[int, Path]]:
    """``[(environment steps, checkpoint path), ...]`` of a training run, sorted by steps."""
    run = Path(run_dir)
    meta = json.loads((run / "train.json").read_text(encoding="utf-8"))
    out: list[tuple[int, Path]] = []
    if meta["library"] == "stable-baselines3":
        for path in (run / "checkpoints").glob("rl_model_*_steps.zip"):
            out.append((int(re.search(r"_(\d+)_steps", path.name).group(1)), path))
    else:
        steps_per_epoch = int(meta["steps_per_epoch"])
        for path in Path(meta["omnisafe_log_dir"]).glob("torch_save/epoch-*.pt"):
            epoch = int(re.search(r"epoch-(\d+)", path.name).group(1))
            if epoch > 0:
                out.append((epoch * steps_per_epoch, path))
    return sorted(out)


def load_predict(run_dir: str | Path, checkpoint: Path, env: Any) -> Predict:
    meta = json.loads((Path(run_dir) / "train.json").read_text(encoding="utf-8"))
    if meta["library"] == "stable-baselines3":
        return _load_sb3(meta, Path(checkpoint), env)
    return _load_omnisafe(meta, Path(checkpoint), env)


def _load_sb3(meta: dict, checkpoint: Path, env: Any) -> Predict:
    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from pv_bess_gym.rl.wrappers import DispatchRLWrapper

    torch.set_num_threads(1)
    steps = re.search(r"_(\d+)_steps", checkpoint.name).group(1)
    stats = checkpoint.with_name(f"rl_model_vecnormalize_{steps}_steps.pkl")
    vecnorm = VecNormalize.load(str(stats), DummyVecEnv([lambda: DispatchRLWrapper(env)]))
    vecnorm.training = False
    model = {"ppo": PPO, "sac": SAC}[meta["algo"]].load(checkpoint, device="cpu")

    def predict(flat_obs: np.ndarray) -> np.ndarray:
        return model.predict(vecnorm.normalize_obs(flat_obs), deterministic=True)[0]

    return predict


def _load_omnisafe(meta: dict, checkpoint: Path, env: Any) -> Predict:
    import torch
    from gymnasium import spaces
    from omnisafe.models.actor import ActorBuilder

    from pv_bess_gym.rl.wrappers import flat_observation_space

    torch.set_num_threads(1)
    config = json.loads((Path(meta["omnisafe_log_dir"]) / "config.json").read_text(encoding="utf-8"))
    model_cfg = config["model_cfgs"]
    obs_space = flat_observation_space(env)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    actor = ActorBuilder(
        obs_space=obs_space, act_space=act_space, hidden_sizes=model_cfg["actor"]["hidden_sizes"],
        activation=model_cfg["actor"]["activation"], weight_initialization_mode=model_cfg["weight_initialization_mode"],
    ).build_actor(model_cfg["actor_type"])
    saved = torch.load(checkpoint, map_location="cpu")
    actor.load_state_dict(saved["pi"])
    actor.eval()
    normalizer = saved.get("obs_normalizer")
    mean = std = None
    clip: Any = 5.0
    if normalizer is not None:
        mean, std = normalizer["_mean"].float(), normalizer["_std"].float()
        if "_clip" in normalizer:
            clip = normalizer["_clip"].float()       # stored per observation entry

    def predict(flat_obs: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(flat_obs, dtype=torch.float32)
        if mean is not None:
            obs = torch.max(torch.min((obs - mean) / std, torch.as_tensor(clip)), -torch.as_tensor(clip))     # same formula as omnisafe's Normalizer, without updating it
        with torch.no_grad():
            action = actor.predict(obs, deterministic=True)
        return np.clip(action.numpy(), -1.0, 1.0)

    return predict

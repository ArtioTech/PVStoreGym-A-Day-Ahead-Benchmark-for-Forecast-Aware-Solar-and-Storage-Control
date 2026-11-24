"""Train and reload Stable-Baselines3 baselines on the dispatch RL task."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from pv_bess_gym.rl.wrappers import DispatchRLWrapper, RLPolicyAgent

ALGO_DEFAULTS: dict[str, dict[str, Any]] = {
    "ppo": dict(n_steps=1920, batch_size=240, n_epochs=10, gamma=0.995, gae_lambda=0.95,
                learning_rate=3e-4, ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128])),
    "sac": dict(buffer_size=200_000, batch_size=256, gamma=0.995, learning_rate=3e-4,
                learning_starts=2_000, train_freq=1, gradient_steps=1, policy_kwargs=dict(net_arch=[128, 128])),
}


def _sb3():
    try:
        import torch
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RL baselines need the 'rl' extra: pip install 'pv_bess_gym[rl]'") from exc
    return torch, {"ppo": PPO, "sac": SAC}, Monitor, DummyVecEnv, VecNormalize


def train_baseline(
    make_env: Callable[[], Any],
    *,
    algo: str,
    projection: bool,
    total_timesteps: int,
    seed: int,
    out_dir: str | Path,
    hyperparameters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train one baseline and save ``model.zip``, ``vecnormalize.pkl``, ``train.json``."""
    torch, algos, Monitor, DummyVecEnv, VecNormalize = _sb3()
    torch.set_num_threads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _factory():
        wrapped = DispatchRLWrapper(make_env(), projection=projection)
        wrapped.reset(seed=seed)
        return Monitor(wrapped, filename=str(out / "monitor"))

    venv = VecNormalize(DummyVecEnv([_factory]), norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)
    params = {**ALGO_DEFAULTS[algo], **(hyperparameters or {})}
    model = algos[algo]("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **params)

    started = time.time()
    model.learn(total_timesteps=int(total_timesteps))
    elapsed = time.time() - started

    model.save(out / "model")
    venv.save(str(out / "vecnormalize.pkl"))
    meta = {
        "algo": algo,
        "projection": projection,
        "seed": seed,
        "total_timesteps": int(total_timesteps),
        "train_seconds": elapsed,
        "steps_per_second": float(total_timesteps) / max(elapsed, 1e-9),
        "hyperparameters": {k: v for k, v in params.items() if k != "policy_kwargs"},
        "net_arch": params["policy_kwargs"]["net_arch"],
        "n_parameters": int(sum(p.numel() for p in model.policy.parameters())),
    }
    (out / "train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_policy_agent(env: Any, run_dir: str | Path, *, algo: str, projection: bool) -> RLPolicyAgent:
    """Rebuild the evaluation agent of a saved run for ``env``."""
    torch, algos, _, DummyVecEnv, VecNormalize = _sb3()
    torch.set_num_threads(1)
    run = Path(run_dir)
    dummy = DummyVecEnv([lambda: DispatchRLWrapper(env, projection=projection)])
    vecnorm = VecNormalize.load(str(run / "vecnormalize.pkl"), dummy)
    vecnorm.training = False
    model = algos[algo].load(run / "model", device="cpu")

    def predict(flat_obs: np.ndarray) -> np.ndarray:
        return model.predict(vecnorm.normalize_obs(flat_obs), deterministic=True)[0]

    return RLPolicyAgent(env, predict, projection=projection)

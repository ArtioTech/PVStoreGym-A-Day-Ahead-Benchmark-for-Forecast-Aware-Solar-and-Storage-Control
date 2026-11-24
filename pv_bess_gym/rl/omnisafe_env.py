"""OmniSafe adapter for the dispatch task (used by PPO-Lagrangian, unmodified).

OmniSafe needs its own environment class. This one exposes the same task the
Stable-Baselines3 baselines train on (:func:`pv_bess_gym.rl.DispatchRLWrapper`)
in constrained form:

- reward: step profit **without** the penalty term, divided by ``reward_scale``;
- cost: limit-violation plus shortfall magnitude of the step, in kW.

Importing this module registers the IDs ``PVBESS-<task>-Dispatch-v2`` with
OmniSafe. Settings that OmniSafe cannot pass through (training days,
activation seeds, the author's c_thr) are read from :data:`CONFIG`, which the
training script fills before it creates the agent.

Requires the OmniSafe environment (Python 3.9 or 3.10); see docs/INSTALL.md.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

import numpy as np
import torch
from gymnasium import spaces
from omnisafe.envs.core import CMDP, env_register

from pv_bess_gym.benchmark.tasks import TASK_NAMES, load_protocol, make_task_env, split_days
from pv_bess_gym.rl.wrappers import DispatchRLWrapper

CONFIG: dict[str, Any] = {
    "split": "train",            # days sampled at reset
    "c_thr_eur_mwh": None,       # needed for T4/T5
    "reward_scale": 10.0,
    "random_soc": True,
    "activation_seeds": None,    # None -> the protocol's seeds
}


def dispatch_env_id(task: str) -> str:
    return f"PVBESS-{task}-Dispatch-v2"


@env_register
class PVBESSDispatchCMDP(CMDP):
    """Single-environment CMDP around ``DispatchRLWrapper(cost_mode=True)``."""

    _support_envs: ClassVar[list[str]] = [dispatch_env_id(name) for name in TASK_NAMES]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False

    def __init__(self, env_id: str, num_envs: int = 1, device: torch.device = torch.device("cpu"), **kwargs: Any) -> None:
        if num_envs != 1:
            raise NotImplementedError("Only one environment per process is supported.")
        self._num_envs = 1
        self._device = torch.device(device)
        task = env_id[len("PVBESS-"):-len("-Dispatch-v2")]
        protocol = load_protocol()
        base = make_task_env(task, c_thr_eur_mwh=CONFIG["c_thr_eur_mwh"])
        base.set_episode_days(split_days(base, CONFIG["split"], protocol))
        seeds = CONFIG["activation_seeds"] or protocol["episode"]["activation_seeds"]
        self._env = DispatchRLWrapper(
            base, cost_mode=True, random_soc=bool(CONFIG["random_soc"]),
            reward_scale=float(CONFIG["reward_scale"]), activation_seeds=seeds,
        )
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self._observation_space = self._env.observation_space
        self._metadata: dict[str, Any] = {}
        self._seed: Optional[int] = None

    def _tensor(self, value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self._device)

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, info = self._env.step(action.detach().cpu().numpy())
        out_info = {key: self._tensor(value) for key, value in info.items()}
        return (
            self._tensor(obs), self._tensor(reward), self._tensor(info["cost"]),
            self._tensor(terminated, torch.bool), self._tensor(truncated, torch.bool), out_info,
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is None and self._seed is not None:
            seed, self._seed = self._seed, None
        obs, info = self._env.reset(seed=seed, options=options)
        return self._tensor(obs), {}

    def set_seed(self, seed: int) -> None:
        self._seed = int(seed)

    @property
    def max_episode_steps(self) -> int:
        return int(self._env.base.episode_len)

    def render(self) -> Any:
        return None

    def close(self) -> None:
        return None

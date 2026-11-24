"""Wrappers that turn the unified environment into a standard RL task.

The learning agent controls the two fast layers, intraday trading and
real-time battery dispatch. The D-1 layer (DAM schedule and aFRR capacity)
comes from the same day-ahead optimiser the DA-OPT baselines use, so the RL
baselines are compared with them on exactly the intraday/real-time problem.

Three pieces are shared by training and evaluation:

``flatten_observation``
    Dict observation -> flat float32 vector with fixed physical scales
    (powers by the PCS rating, prices by 100 EUR/MWh, ...).
``scale_action``
    Normalised action in [-1, 1]^2 -> ``(id_adjustment_kw, p_batt_kw)``.
``project_setpoint``
    Optional safety layer: projects the battery setpoint onto the set that
    respects the PCS rating net of the reserved aFRR capacity and the energy
    available between the SOC limits. This is the constraint-aware variant.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pv_bess_gym.agents.day_ahead_baselines import DayAheadMPCConfig, DayAheadOptimizedDARBCAgent

PRICE_SCALE = 100.0          # EUR/MWh
CAP_PRICE_SCALE = 20.0       # EUR/MW/h
N_PHYSICAL = 24
FORECAST_COLS = 9


def _scales(env: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-entry divisors for ``physical`` and one ``forecast`` row."""
    p = float(env.plant.battery.p_pcs_kw)
    pv = float(env.plant.pv.p_ac_kw)
    e = float(env.plant.battery.e_max_kwh)
    physical = np.array(
        [
            1.0, e, pv, pv, 1000.0, 40.0, 10.0,
            PRICE_SCALE, PRICE_SCALE, PRICE_SCALE, CAP_PRICE_SCALE, CAP_PRICE_SCALE,
            1.0, 1.0, p, p, p,
            max(1.0, float(env.intraday_block_steps)), max(1.0, float(env.afrr_block_steps)),
            1.0, 1.0, 1.0, p, p,
        ],
        dtype=np.float32,
    )
    forecast = np.array(
        [pv, 1000.0, 40.0, PRICE_SCALE, PRICE_SCALE, PRICE_SCALE, CAP_PRICE_SCALE, CAP_PRICE_SCALE, p],
        dtype=np.float32,
    )
    return physical, forecast


def flatten_observation(env: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
    """Fixed-scale, flat observation: ``physical`` + ``forecast`` + intraday mask."""
    physical_scale, forecast_scale = _scales(env)
    physical = np.asarray(obs["physical"], dtype=np.float32) / physical_scale
    forecast = np.asarray(obs["forecast"], dtype=np.float32).reshape(-1, FORECAST_COLS) / forecast_scale
    mask = np.asarray(obs["action_mask"], dtype=np.float32)[1:2]
    return np.clip(np.concatenate([physical, forecast.reshape(-1), mask]), -10.0, 10.0).astype(np.float32)


def flat_observation_space(env: Any) -> spaces.Box:
    dim = N_PHYSICAL + env.forecast_horizon_steps * FORECAST_COLS + 1
    return spaces.Box(low=-10.0, high=10.0, shape=(dim,), dtype=np.float32)


def scale_action(env: Any, action: np.ndarray) -> tuple[float, float]:
    """Map a normalised action in [-1, 1]^2 to (id_adjustment_kw, p_batt_kw)."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    max_sched = max(env.plant.grid.export_cap_kw, env.plant.grid.import_cap_kw)
    return float(a[0] * max_sched), float(a[1] * env.plant.battery.p_pcs_kw)


def project_setpoint(env: Any, p_batt_kw: float, soc: float, cap_up_kw: float, cap_down_kw: float) -> float:
    """Project a battery setpoint onto the feasible set (safety layer).

    Feasible means that the setpoint plus *full* activation of the reserved
    aFRR capacity, in either direction, stays within the PCS rating and within
    what the battery can deliver at this SOC (taper and available energy). A
    projected setpoint therefore causes neither a reserve violation nor a
    delivery shortfall, whatever the activation of the step.
    """
    from pv_bess_gym.models.battery import Battery

    cfg, opts = env.plant.battery, env.options
    probe = Battery(
        cfg, soc_init=soc, aux_drains=opts.battery_aux_drains, directional_taper=opts.directional_taper,
        energy_limited_power=opts.energy_limited_power,
    )
    dis_lim, ch_lim = probe.directional_limits_kw()
    if opts.energy_limited_power:
        dis_energy, ch_energy = probe.energy_limits_kw(float(env.market.dt_h))
        dis_lim, ch_lim = min(dis_lim, dis_energy), min(ch_lim, ch_energy)
    guard = 1e-3  # 1 W, so float32 rounding of the action cannot cross a limit
    max_dis = max(0.0, min(cfg.p_pcs_kw, dis_lim) - cap_up_kw - guard)
    max_ch = max(0.0, min(cfg.p_pcs_kw, ch_lim) - cap_down_kw - guard)
    return float(np.clip(p_batt_kw, -max_ch, max_dis))


class _D1Planner:
    """The shared day-ahead optimiser of the DA-OPT baselines."""

    def __init__(self, env: Any, config: Optional[DayAheadMPCConfig] = None):
        self._agent = DayAheadOptimizedDARBCAgent(env, config=config)

    def plan(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        self._agent.reset()
        return self._agent._build_d1_plan(obs)


def _build_action(plan: dict[str, Any], id_kw: float, p_batt_kw: float) -> dict[str, np.ndarray]:
    return {
        "day_ahead": np.asarray(plan["schedule"], dtype=np.float32),
        "day_ahead_afrr": np.array([plan["cap_up"], plan["cap_down"]], dtype=np.float32),
        "intraday": np.array([id_kw], dtype=np.float32),
        "real_time": np.array([p_batt_kw], dtype=np.float32),
    }


class FixedD1Wrapper(gym.Wrapper):
    """Wrapper (i): fixes the D-1 action to a supplied D-1 policy.

    The D-1 schedule and aFRR capacities come from ``d1_policy(obs) -> plan``
    (default: the shared day-ahead optimisation of the DA-OPT baselines). The
    remaining action is ``{"intraday": [kW], "real_time": [kW]}``; observations
    are passed through unchanged.
    """

    def __init__(self, env: Any, d1_config: Optional[DayAheadMPCConfig] = None, d1_policy: Any = None):
        super().__init__(env)
        base = env.unwrapped
        self._d1_policy = d1_policy or _D1Planner(base, d1_config).plan
        self.plan: dict[str, Any] = {}
        self.last_obs: dict[str, np.ndarray] = {}
        self.action_space = spaces.Dict(
            {"intraday": base.action_space["intraday"], "real_time": base.action_space["real_time"]}
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        self.last_obs, info = self.env.reset(seed=seed, options=options)
        self.plan = self._d1_policy(self.last_obs)
        return self.last_obs, info

    def step(self, action: dict[str, Any]):
        id_kw = float(np.asarray(action["intraday"], dtype=float).reshape(-1)[0])
        p_kw = float(np.asarray(action["real_time"], dtype=float).reshape(-1)[0])
        self.last_obs, reward, terminated, truncated, info = self.env.step(_build_action(self.plan, id_kw, p_kw))
        return self.last_obs, reward, terminated, truncated, info


class FlatDispatchWrapper(gym.Wrapper):
    """Wrapper (ii): flattens the remaining keys to ``(y, b)`` in [-1, 1]^2 and the observation.

    Must wrap a :class:`FixedD1Wrapper`. The intraday component ``y`` is
    ignored by the environment at steps where no intraday decision is due.

    - action: Box(-1, 1, (2,)) = normalised (intraday adjustment, battery setpoint)
    - observation: flat, fixed-scale vector (see :func:`flatten_observation`)
    - reward: step profit in EUR divided by ``reward_scale``
    - ``cost_mode=True`` (constrained RL): the reward excludes the penalty term
      and ``info["cost"]`` is the limit-violation plus shortfall magnitude of
      the step in kW
    - training conveniences: each ``reset`` samples a day from the env's
      ``episode_days``; ``random_soc`` samples the initial SOC;
      ``activation_seeds`` samples an activation seed per episode
    - ``projection=True`` is the optional safety layer of :func:`project_setpoint`
    """

    def __init__(
        self,
        env: FixedD1Wrapper,
        *,
        projection: bool = False,
        random_soc: bool = True,
        reward_scale: float = 10.0,
        cost_mode: bool = False,
        activation_seeds: Optional[Sequence[int]] = None,
    ):
        if not isinstance(env, FixedD1Wrapper):
            raise TypeError("FlatDispatchWrapper must wrap a FixedD1Wrapper.")
        super().__init__(env)
        self.base = env.unwrapped
        self.projection = bool(projection)
        self.random_soc = bool(random_soc)
        self.reward_scale = float(reward_scale)
        self.cost_mode = bool(cost_mode)
        self.activation_seeds = None if activation_seeds is None else [int(s) for s in activation_seeds]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = flat_observation_space(self.base)
        self._rng = np.random.default_rng()   # episode sampling (day, SOC, activation seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        options = dict(options or {})
        if "day" not in options and "start_idx" not in options:
            days = self.base.episode_days
            options["day"] = int(days[int(rng.integers(len(days)))])
        if self.random_soc and "soc_init" not in options:
            cfg = self.base.plant.battery
            options["soc_init"] = float(rng.uniform(cfg.soc_min + 0.05, cfg.soc_max - 0.05))
        if self.activation_seeds is not None and "activation_seed" not in options:
            options["activation_seed"] = int(self.activation_seeds[int(rng.integers(len(self.activation_seeds)))])
        obs, info = self.env.reset(seed=seed, options=options)
        return flatten_observation(self.base, obs), info

    def step(self, action: np.ndarray):
        id_kw, p_batt_kw = scale_action(self.base, action)
        if self.projection:
            plan = self.env.plan
            p_batt_kw = project_setpoint(
                self.base, p_batt_kw, float(self.env.last_obs["physical"][0]), plan["cap_up"], plan["cap_down"]
            )
        obs, reward, terminated, truncated, full = self.env.step(
            {"intraday": np.array([id_kw], dtype=np.float32), "real_time": np.array([p_batt_kw], dtype=np.float32)}
        )
        shortfall_kw = float(full["shortfall_mwh"]) * 1000.0 / float(self.base.market.dt_h)
        cost = float(full["violation_kw"]) + shortfall_kw
        info = {
            "profit_eur": float(reward),
            "penalties_eur": float(full["penalties_eur"]),
            "violation_kw": float(full["violation_kw"]),
            "shortfall_mwh": float(full["shortfall_mwh"]),
            "cost": cost,
        }
        learn_reward = float(reward) + (float(full["penalties_eur"]) if self.cost_mode else 0.0)
        return flatten_observation(self.base, obs), learn_reward / self.reward_scale, terminated, truncated, info


def DispatchRLWrapper(env: Any, *, d1_config: Optional[DayAheadMPCConfig] = None, **kwargs: Any) -> FlatDispatchWrapper:
    """Both wrappers applied: ``FlatDispatchWrapper(FixedD1Wrapper(env), **kwargs)``."""
    return FlatDispatchWrapper(FixedD1Wrapper(env, d1_config=d1_config), **kwargs)


class RLPolicyAgent:
    """Adapter so a trained policy runs in ``evaluate_agent`` like any controller.

    ``predict`` maps a flat observation to a normalised action; pass
    ``lambda o: model.predict(vecnorm.normalize_obs(o), deterministic=True)[0]``
    for a Stable-Baselines3 model trained with ``VecNormalize``.
    """

    def __init__(self, env: Any, predict, *, projection: bool = False, d1_config: Optional[DayAheadMPCConfig] = None):
        self.env = env
        self.predict = predict
        self.projection = bool(projection)
        self._planner = _D1Planner(env, d1_config)
        self._plan: Optional[dict[str, Any]] = None

    def reset(self) -> None:
        self._plan = None

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._plan is None:
            self._plan = self._planner.plan(obs)
        action = np.asarray(self.predict(flatten_observation(self.env, obs)), dtype=np.float64).reshape(-1)
        id_kw, p_batt_kw = scale_action(self.env, action)
        if self.projection:
            p_batt_kw = project_setpoint(
                self.env, p_batt_kw, float(obs["physical"][0]), self._plan["cap_up"], self._plan["cap_down"]
            )
        return _build_action(self._plan, id_kw, p_batt_kw)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from pv_bess_gym.models.battery import BatteryConfig
from pv_bess_gym.models.grid import GridConfig

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - fallback path
    cp = None


@dataclass
class DayAheadMPCConfig:
    """Tuning for the Greek-market-sequenced day-ahead benchmark agents.

    The day-ahead schedule is intentionally optimized for every benchmark.
    This lets experiments compare intraday and real-time operators under the
    same upper-level commercial commitment:
    - DA-OPT + RBC/RBC: optimized DAM schedule, rule-based intraday/reserve and
      rule-based real-time tracking.
    - DA-OPT + MPC/MPC: the same optimized DAM schedule, MPC intraday/reserve
      corrections, and MPC real-time battery dispatch.
    """

    day_ahead_confidence_factor: float = 0.90
    day_ahead_tracking_penalty_eur_mwh: float = 160.0
    intraday_confidence_factor: float = 0.98
    tracking_gain: float = 0.90
    reserve_fraction: float = 0.20
    reserve_max_fraction: float = 0.50
    reserve_price_threshold: float = 20.0
    reserve_soc_buffer_fraction: float = 0.08
    reserve_sustain_hours: float = 0.50
    low_price_eur_mwh: float = 45.0
    high_price_eur_mwh: float = 95.0
    price_bias_fraction: float = 0.08
    realtime_horizon_steps: int = 8
    intraday_mpc_horizon_steps: int = 8
    imbalance_penalty_eur_mwh: float = 140.0
    throughput_penalty_eur_mwh: float = 2.5
    terminal_soc_target: float = 0.55
    terminal_soc_weight: float = 12.0
    solver: str = "CLARABEL"
    solver_max_iter: int = 20000


class DayAheadOptimizedDARBCAgent:
    """Optimized day-ahead schedule with RBC intraday and real-time operation.

    This is the first clean benchmark. The upper-upper market layer solves a
    deterministic full-day schedule using forecast PV, DAM prices, battery
    limits, grid limits, and terminal SOC regularization. After this common
    optimized schedule is fixed, intraday/reserve and real-time operation use
    transparent rules.

    The agent receives the full delivery-day scheduling context through the Gym
    observation. It keeps a reference to the environment only for static plant,
    action-space, and timing parameters needed by this deterministic benchmark.
    """

    name = "day_ahead_optimized_da_rbc"

    def __init__(self, env: Any, config: Optional[DayAheadMPCConfig] = None):
        self.env = env
        self.action_space = env.action_space
        self.battery_cfg: BatteryConfig = env.plant.battery
        self.grid_cfg: GridConfig = env.plant.grid
        self.dt_hours = float(env.market.dt_h)
        self.config = config or DayAheadMPCConfig()

        self.eta = float(np.clip(self.battery_cfg.eta_nom, 0.85, 0.995))
        self.self_discharge_factor = 1.0 - self.battery_cfg.self_discharge_per_day * (self.dt_hours / 24.0)
        self.reserve_power_cap_kw = self.config.reserve_max_fraction * self.battery_cfg.p_pcs_kw
        self.reserve_soc_buffer_kwh = self.config.reserve_soc_buffer_fraction * self.battery_cfg.e_max_kwh
        self.reset()

    def reset(self) -> None:
        self._day_ahead_schedule: Optional[np.ndarray] = None
        self._d1_cap_up_kw: float = 0.0
        self._d1_cap_down_kw: float = 0.0
        self._last_intraday = np.zeros(1, dtype=np.float32)

    def _forecast_chunks(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        forecast = np.asarray(obs.get("forecast", np.array([], dtype=np.float32)), dtype=float)
        if forecast.size == 0:
            return np.zeros((0, 9), dtype=float)
        return forecast.reshape(-1, 9)

    def _day_ahead_context_chunks(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        context = np.asarray(obs.get("day_ahead_context", np.array([], dtype=np.float32)), dtype=float)
        if context.size == 0:
            return np.zeros((0, 8), dtype=float)
        return context.reshape(-1, 8)

    def _clip_day_ahead(self, schedule_kw: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(schedule_kw, dtype=np.float32),
            self.action_space["day_ahead"].low,
            self.action_space["day_ahead"].high,
        )

    def _clip_intraday(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space["intraday"].low,
            self.action_space["intraday"].high,
        )

    def _clip_low_level(self, p_batt_kw: float) -> np.ndarray:
        action_spaces = getattr(self.action_space, "spaces", {})
        space_key = "real_time" if "real_time" in action_spaces else "low_level"
        return np.clip(
            np.array([p_batt_kw], dtype=np.float32),
            self.action_space[space_key].low,
            self.action_space[space_key].high,
        )

    def _blank_action(self) -> dict[str, np.ndarray]:
        """Return a full Gym action dictionary with inactive blocks zeroed."""
        return {
            "day_ahead": np.zeros(int(self.env.episode_len), dtype=np.float32),
            "day_ahead_afrr": np.zeros(2, dtype=np.float32),
            "intraday": np.zeros(1, dtype=np.float32),
            "real_time": np.zeros(1, dtype=np.float32),
        }

    def _episode_forecast_pv_kw(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        context = self._day_ahead_context_chunks(obs)
        if context.size:
            out = np.zeros(int(self.env.episode_len), dtype=float)
            take = min(out.size, context.shape[0])
            out[:take] = context[:take, 0]
            if take < out.size and take > 0:
                out[take:] = out[take - 1]
            return out

        episode_df = getattr(self.env, "episode_df", None)
        if episode_df is not None:
            values = [self.env._weather_power(row, forecast=True) for _, row in episode_df.iterrows()]
            return np.asarray(values, dtype=float)

        chunks = self._forecast_chunks(obs)
        if chunks.size == 0:
            return np.zeros(self.env.episode_len, dtype=float)
        return np.resize(chunks[:, 0], self.env.episode_len).astype(float)

    def _episode_dam_prices(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        context = self._day_ahead_context_chunks(obs)
        if context.size:
            out = np.zeros(int(self.env.episode_len), dtype=float)
            take = min(out.size, context.shape[0])
            out[:take] = context[:take, 1]
            if take < out.size and take > 0:
                out[take:] = out[take - 1]
            return out

        episode_df = getattr(self.env, "episode_df", None)
        schema = getattr(self.env, "schema", None)
        if episode_df is not None and schema is not None:
            return episode_df[schema.price_dam_eur_mwh].to_numpy(dtype=float)

        chunks = self._forecast_chunks(obs)
        if chunks.size == 0:
            return np.full(self.env.episode_len, float(obs["physical"][7]), dtype=float)
        return np.resize(chunks[:, 3], self.env.episode_len).astype(float)

    def _fallback_day_ahead_schedule(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        pv_fcst = self._episode_forecast_pv_kw(obs)
        dam_price = self._episode_dam_prices(obs)

        schedule = self.config.day_ahead_confidence_factor * pv_fcst
        schedule = np.where(dam_price >= 0.0, schedule, 0.0)
        schedule = np.clip(schedule, 0.0, self.grid_cfg.export_cap_kw)
        return self._clip_day_ahead(schedule)

    def _solve_d1_cooptimization(self, obs: dict[str, np.ndarray]) -> Optional[dict[str, Any]]:
        """Solve D-1 co-optimization of DAM schedule and aFRR capacity.

        This deterministic optimizer jointly decides the DAM energy schedule and
        the aFRR capacity offer (upward and downward). Both are committed at D-1
        for the full delivery day, correctly reflecting the Greek market structure.

        Returns dict with keys: schedule, cap_up, cap_down. Returns None on failure.
        """
        if cp is None:
            return None

        pv_fcst = self._episode_forecast_pv_kw(obs)
        dam_price = self._episode_dam_prices(obs)
        afrr_up_price, afrr_down_price = self._episode_afrr_cap_prices(obs)
        n = min(len(pv_fcst), len(dam_price), len(afrr_up_price), len(afrr_down_price), int(self.env.episode_len))
        if n <= 0:
            return None

        pv_fcst = pv_fcst[:n]
        dam_price = dam_price[:n]
        afrr_up_price = afrr_up_price[:n]
        afrr_down_price = afrr_down_price[:n]

        schedule = cp.Variable(n)
        cap_up_d1 = cp.Variable(nonneg=True)
        cap_down_d1 = cp.Variable(nonneg=True)
        p_dis = cp.Variable(n, nonneg=True)
        p_ch = cp.Variable(n, nonneg=True)
        imbalance_abs = cp.Variable(n, nonneg=True)
        energy = cp.Variable(n + 1)

        e_min = self.battery_cfg.soc_min * self.battery_cfg.e_max_kwh
        e_max = self.battery_cfg.soc_max * self.battery_cfg.e_max_kwh
        e_init = float(
            np.clip(float(obs["physical"][0]), self.battery_cfg.soc_min, self.battery_cfg.soc_max)
            * self.battery_cfg.e_max_kwh
        )
        sustain_h = max(self.dt_hours, self.config.reserve_sustain_hours)

        constraints = [
            energy[0] == e_init,
            energy >= e_min,
            energy <= e_max,
            schedule >= 0.0,
            schedule <= self.grid_cfg.export_cap_kw,
            cap_up_d1 <= self.reserve_power_cap_kw,
            cap_down_d1 <= self.reserve_power_cap_kw,
            cap_up_d1 + cap_down_d1 <= self.battery_cfg.p_pcs_kw,
        ]
        revenue_terms = []
        penalty_terms = []

        for t in range(n):
            net_grid_kw = pv_fcst[t] + p_dis[t] - p_ch[t]
            constraints.extend(
                [
                    p_dis[t] + cap_up_d1 <= self.battery_cfg.p_pcs_kw,
                    p_ch[t] + cap_down_d1 <= self.battery_cfg.p_pcs_kw,
                    net_grid_kw <= self.grid_cfg.export_cap_kw,
                    net_grid_kw >= -self.grid_cfg.import_cap_kw,
                    energy[t] >= e_min + self.reserve_soc_buffer_kwh + (sustain_h / self.eta) * cap_up_d1,
                    energy[t] <= e_max - self.reserve_soc_buffer_kwh - (sustain_h * self.eta) * cap_down_d1,
                    energy[t + 1]
                    == self.self_discharge_factor * energy[t]
                    - (self.dt_hours / self.eta) * p_dis[t]
                    + (self.dt_hours * self.eta) * p_ch[t],
                    imbalance_abs[t] >= net_grid_kw - schedule[t],
                    imbalance_abs[t] >= -(net_grid_kw - schedule[t]),
                ]
            )
            revenue_terms.append(float(dam_price[t]) * schedule[t] * (self.dt_hours / 1000.0))
            revenue_terms.append(float(afrr_up_price[t]) * cap_up_d1 * (self.dt_hours / 1000.0))
            revenue_terms.append(float(afrr_down_price[t]) * cap_down_d1 * (self.dt_hours / 1000.0))
            penalty_terms.append(
                self.config.day_ahead_tracking_penalty_eur_mwh
                * imbalance_abs[t]
                * (self.dt_hours / 1000.0)
            )
            penalty_terms.append(
                self.config.throughput_penalty_eur_mwh
                * (p_dis[t] + p_ch[t])
                * (self.dt_hours / 1000.0)
            )

        target_energy = self.config.terminal_soc_target * self.battery_cfg.e_max_kwh
        terminal_penalty = self.config.terminal_soc_weight * cp.square(
            (energy[-1] - target_energy) / self.battery_cfg.e_max_kwh
        )
        problem = cp.Problem(cp.Maximize(cp.sum(revenue_terms) - cp.sum(penalty_terms) - terminal_penalty), constraints)

        try:
            problem.solve(solver=self.config.solver, warm_start=True, max_iter=self.config.solver_max_iter, verbose=False)
        except Exception:
            return None

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or schedule.value is None:
            return None

        out_schedule = np.zeros(int(self.env.episode_len), dtype=np.float32)
        out_schedule[:n] = np.asarray(schedule.value, dtype=np.float32)
        if n < int(self.env.episode_len):
            out_schedule[n:] = out_schedule[n - 1]

        return {
            "schedule": self._clip_day_ahead(out_schedule),
            "cap_up": float(np.clip(cap_up_d1.value or 0.0, 0.0, self.reserve_power_cap_kw)),
            "cap_down": float(np.clip(cap_down_d1.value or 0.0, 0.0, self.reserve_power_cap_kw)),
        }

    def _build_d1_plan(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        """Build complete D-1 plan: DAM schedule and aFRR capacity."""
        result = self._solve_d1_cooptimization(obs)
        if result is not None:
            return result
        schedule = self._fallback_day_ahead_schedule(obs)
        cap_up, cap_down = self._rbc_reserve_offer(obs)
        return {"schedule": schedule, "cap_up": cap_up, "cap_down": cap_down}

    def _build_day_ahead_schedule(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Legacy alias for backwards compatibility."""
        return self._build_d1_plan(obs)["schedule"]

    def _target_slice(self) -> slice:
        start = min(self.env.episode_len, int(self.env.t) + int(getattr(self.env, "intraday_gate_steps", 0)))
        block_steps = int(getattr(self.env, "intraday_block_steps", 1))
        stop = min(self.env.episode_len, start + block_steps)
        return slice(start, stop)

    def _pv_forecast_for_slice(self, target: slice, obs: dict[str, np.ndarray]) -> np.ndarray:
        context = self._day_ahead_context_chunks(obs)
        if context.size and target.start < target.stop:
            take = min(target.stop, context.shape[0])
            if target.start < take:
                return context[target.start:take, 0].astype(float)

        chunks = self._forecast_chunks(obs)
        if chunks.size and target.start < target.stop:
            rel_start = max(1, target.start - int(getattr(self.env, "t", 0)))
            rel_stop = max(rel_start, target.stop - int(getattr(self.env, "t", 0)))
            start_idx = rel_start - 1
            stop_idx = min(chunks.shape[0], rel_stop - 1)
            if start_idx < stop_idx:
                return chunks[start_idx:stop_idx, 0].astype(float)

        episode_df = getattr(self.env, "episode_df", None)
        if episode_df is not None and target.start < target.stop:
            rows = episode_df.iloc[target]
            return np.asarray([self.env._weather_power(row, forecast=True) for _, row in rows.iterrows()], dtype=float)

        if chunks.size == 0:
            return np.array([float(obs["physical"][3])], dtype=float)
        return chunks[: max(1, target.stop - target.start), 0].astype(float)

    def _mean_profile(self, profile: np.ndarray, target: slice, fallback: float) -> float:
        if target.start >= target.stop or profile.size == 0:
            return float(fallback)
        return float(np.mean(profile[target]))

    def _reserve_energy_limits(self, soc: float) -> tuple[float, float]:
        e_min = self.battery_cfg.soc_min * self.battery_cfg.e_max_kwh
        e_max = self.battery_cfg.soc_max * self.battery_cfg.e_max_kwh
        e_now = float(np.clip(soc, self.battery_cfg.soc_min, self.battery_cfg.soc_max) * self.battery_cfg.e_max_kwh)
        sustain_h = max(self.dt_hours, self.config.reserve_sustain_hours)

        up_kwh = max(0.0, e_now - e_min - self.reserve_soc_buffer_kwh)
        down_kwh = max(0.0, e_max - e_now - self.reserve_soc_buffer_kwh)

        up_kw = up_kwh * self.eta / sustain_h
        down_kw = down_kwh / max(self.eta * sustain_h, 1e-6)
        return min(self.reserve_power_cap_kw, up_kw), min(self.reserve_power_cap_kw, down_kw)

    def _episode_afrr_cap_prices(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Extract aFRR capacity price vectors from day_ahead_context (columns 4 and 5)."""
        context = self._day_ahead_context_chunks(obs)
        n = int(self.env.episode_len)

        if context.size and context.shape[0] >= n:
            return context[:n, 4].astype(float), context[:n, 5].astype(float)

        # Fallback: use physical observation values repeated
        physical = np.asarray(obs["physical"], dtype=float)
        return (
            np.full(n, float(physical[10]), dtype=float),
            np.full(n, float(physical[11]), dtype=float),
        )

    def _rbc_reserve_offer(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        physical = np.asarray(obs["physical"], dtype=float)
        soc = float(physical[0])
        cap_up_price = float(physical[10])
        cap_down_price = float(physical[11])
        reserve_up_limit_kw, reserve_down_limit_kw = self._reserve_energy_limits(soc)
        nominal_reserve_kw = self.config.reserve_fraction * self.battery_cfg.p_pcs_kw

        cap_up = nominal_reserve_kw if (cap_up_price >= self.config.reserve_price_threshold and soc > 0.35) else 0.0
        cap_down = nominal_reserve_kw if (cap_down_price >= self.config.reserve_price_threshold and soc < 0.75) else 0.0
        return min(cap_up, reserve_up_limit_kw), min(cap_down, reserve_down_limit_kw)

    def _rbc_intraday_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        if self._day_ahead_schedule is None:
            self._day_ahead_schedule = self._build_day_ahead_schedule(obs)

        target = self._target_slice()
        pv_target = self._pv_forecast_for_slice(target, obs)
        forecast_target_kw = float(np.mean(pv_target)) if pv_target.size else float(obs["physical"][3])
        dam_target_kw = self._mean_profile(
            self._day_ahead_schedule,
            target,
            fallback=float(obs["physical"][14]),
        )
        desired_final_kw = self.config.intraday_confidence_factor * min(forecast_target_kw, self.grid_cfg.export_cap_kw)
        id_adjustment_kw = desired_final_kw - dam_target_kw

        return self._clip_intraday(np.array([id_adjustment_kw], dtype=np.float32))

    def _soc_limited_dispatch_bounds(self, cap_up_kw: float, cap_down_kw: float, soc: float) -> tuple[float, float]:
        max_discharge_kw = max(0.0, self.battery_cfg.p_pcs_kw - cap_up_kw)
        max_charge_kw = max(0.0, self.battery_cfg.p_pcs_kw - cap_down_kw)
        if soc <= self.battery_cfg.soc_min + 0.02:
            max_discharge_kw = 0.0
        if soc >= self.battery_cfg.soc_max - 0.02:
            max_charge_kw = 0.0
        return -max_charge_kw, max_discharge_kw

    def _rbc_realtime_dispatch_kw(self, obs: dict[str, np.ndarray]) -> float:
        physical = np.asarray(obs["physical"], dtype=float)
        soc = float(physical[0])
        pv_now = float(physical[2])
        price_dam = float(physical[7])
        scheduled_kw = float(physical[16])

        p_batt_kw = self.config.tracking_gain * (scheduled_kw - pv_now)
        if price_dam <= self.config.low_price_eur_mwh and soc < 0.85:
            p_batt_kw -= self.config.price_bias_fraction * self.battery_cfg.p_pcs_kw
        elif price_dam >= self.config.high_price_eur_mwh and soc > 0.20:
            p_batt_kw += self.config.price_bias_fraction * self.battery_cfg.p_pcs_kw

        lower, upper = self._soc_limited_dispatch_bounds(self._d1_cap_up_kw, self._d1_cap_down_kw, soc)
        return float(np.clip(p_batt_kw, lower, upper))

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        action_mask = np.asarray(obs["action_mask"], dtype=float)
        action = self._blank_action()

        if action_mask[0] > 0.5:
            plan = self._build_d1_plan(obs)
            self._day_ahead_schedule = plan["schedule"]
            self._d1_cap_up_kw = float(plan["cap_up"])
            self._d1_cap_down_kw = float(plan["cap_down"])
            action["day_ahead"] = self._day_ahead_schedule.astype(np.float32)
            action["day_ahead_afrr"] = np.array([self._d1_cap_up_kw, self._d1_cap_down_kw], dtype=np.float32)

        if action_mask[1] > 0.5:
            self._last_intraday = self._rbc_intraday_action(obs).astype(np.float32)
            action["intraday"] = self._last_intraday.copy()

        action["real_time"] = self._clip_low_level(self._rbc_realtime_dispatch_kw(obs))
        return action


class DayAheadOptimizedDARTMPCAgent(DayAheadOptimizedDARBCAgent):
    """Ablation: optimized day-ahead, RBC intraday/reserve, MPC real-time."""

    name = "day_ahead_optimized_da_rt_mpc"

    def _build_realtime_horizon(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        physical = np.asarray(obs["physical"], dtype=float)
        chunks = self._forecast_chunks(obs)
        horizon = max(1, min(self.config.realtime_horizon_steps, 1 + chunks.shape[0]))

        pv = np.full(horizon, float(physical[2]), dtype=float)
        scheduled = np.full(horizon, float(physical[16]), dtype=float)
        imb_price = np.full(horizon, float(physical[9]), dtype=float)

        if horizon > 1 and chunks.size:
            take = min(horizon - 1, chunks.shape[0])
            pv[1:1 + take] = chunks[:take, 0]
            imb_price[1:1 + take] = chunks[:take, 5]
            scheduled[1:1 + take] = chunks[:take, 8]
            if 1 + take < horizon:
                pv[1 + take:] = pv[take]
                imb_price[1 + take:] = imb_price[take]
                scheduled[1 + take:] = scheduled[take]

        return {"pv_kw": pv, "scheduled_kw": scheduled, "imb_price": imb_price}

    def _solve_realtime_mpc(self, obs: dict[str, np.ndarray]) -> Optional[float]:
        if cp is None:
            return None

        physical = np.asarray(obs["physical"], dtype=float)
        horizon = self._build_realtime_horizon(obs)
        pv = horizon["pv_kw"]
        scheduled = horizon["scheduled_kw"]
        imb_price = horizon["imb_price"]
        num_steps = len(pv)

        cap_up_kw = self._d1_cap_up_kw
        cap_down_kw = self._d1_cap_down_kw
        p_dis = cp.Variable(num_steps, nonneg=True)
        p_ch = cp.Variable(num_steps, nonneg=True)
        imbalance_abs = cp.Variable(num_steps, nonneg=True)
        energy = cp.Variable(num_steps + 1)

        e_min = self.battery_cfg.soc_min * self.battery_cfg.e_max_kwh
        e_max = self.battery_cfg.soc_max * self.battery_cfg.e_max_kwh
        e_init = float(np.clip(physical[0], self.battery_cfg.soc_min, self.battery_cfg.soc_max) * self.battery_cfg.e_max_kwh)
        sustain_h = max(self.dt_hours, self.config.reserve_sustain_hours)

        constraints = [energy[0] == e_init, energy >= e_min, energy <= e_max]
        objective_terms = []

        for t in range(num_steps):
            net_grid_kw = pv[t] + p_dis[t] - p_ch[t]
            constraints.extend(
                [
                    p_dis[t] + cap_up_kw <= self.battery_cfg.p_pcs_kw,
                    p_ch[t] + cap_down_kw <= self.battery_cfg.p_pcs_kw,
                    net_grid_kw <= self.grid_cfg.export_cap_kw,
                    net_grid_kw >= -self.grid_cfg.import_cap_kw,
                    energy[t] >= e_min + self.reserve_soc_buffer_kwh + (sustain_h / self.eta) * cap_up_kw,
                    energy[t] <= e_max - self.reserve_soc_buffer_kwh - (sustain_h * self.eta) * cap_down_kw,
                    energy[t + 1] == self.self_discharge_factor * energy[t] - (self.dt_hours / self.eta) * p_dis[t] + (self.dt_hours * self.eta) * p_ch[t],
                    imbalance_abs[t] >= net_grid_kw - scheduled[t],
                    imbalance_abs[t] >= -(net_grid_kw - scheduled[t]),
                ]
            )
            imbalance_weight = max(self.config.imbalance_penalty_eur_mwh, abs(float(imb_price[t])))
            objective_terms.append(imbalance_weight * imbalance_abs[t] * (self.dt_hours / 1000.0))
            objective_terms.append(self.config.throughput_penalty_eur_mwh * (p_dis[t] + p_ch[t]) * (self.dt_hours / 1000.0))

        target_energy = self.config.terminal_soc_target * self.battery_cfg.e_max_kwh
        terminal_penalty = self.config.terminal_soc_weight * cp.square((energy[-1] - target_energy) / self.battery_cfg.e_max_kwh)
        problem = cp.Problem(cp.Minimize(cp.sum(objective_terms) + terminal_penalty), constraints)

        try:
            problem.solve(solver=self.config.solver, warm_start=True, max_iter=self.config.solver_max_iter, verbose=False)
        except Exception:
            return None

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or p_dis.value is None or p_ch.value is None:
            return None
        return float(p_dis.value[0] - p_ch.value[0])

    def _realtime_dispatch_kw(self, obs: dict[str, np.ndarray]) -> float:
        mpc_value = self._solve_realtime_mpc(obs)
        if mpc_value is None:
            return self._rbc_realtime_dispatch_kw(obs)

        soc = float(obs["physical"][0])
        lower, upper = self._soc_limited_dispatch_bounds(
            self._d1_cap_up_kw,
            self._d1_cap_down_kw,
            soc,
        )
        return float(np.clip(mpc_value, lower, upper))

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        action_mask = np.asarray(obs["action_mask"], dtype=float)
        action = self._blank_action()

        if action_mask[0] > 0.5:
            plan = self._build_d1_plan(obs)
            self._day_ahead_schedule = plan["schedule"]
            self._d1_cap_up_kw = float(plan["cap_up"])
            self._d1_cap_down_kw = float(plan["cap_down"])
            action["day_ahead"] = self._day_ahead_schedule.astype(np.float32)
            action["day_ahead_afrr"] = np.array([self._d1_cap_up_kw, self._d1_cap_down_kw], dtype=np.float32)

        if action_mask[1] > 0.5:
            self._last_intraday = self._rbc_intraday_action(obs).astype(np.float32)
            action["intraday"] = self._last_intraday.copy()

        action["real_time"] = self._clip_low_level(self._realtime_dispatch_kw(obs))
        return action


class DayAheadOptimizedDAMPCIntradayRealTimeAgent(DayAheadOptimizedDARTMPCAgent):
    """Optimized day-ahead schedule with MPC intraday/reserve and real time."""

    name = "day_ahead_optimized_da_mpc_id_rt"

    def _slice_prices(self, target: slice, obs: dict[str, np.ndarray], price_col: str) -> np.ndarray:
        schema = getattr(self.env, "schema", None)
        context = self._day_ahead_context_chunks(obs)
        if context.size and target.start < target.stop and schema is not None:
            if price_col == schema.price_id_eur_mwh:
                col_idx = 2
            elif price_col == schema.price_imb_eur_mwh:
                col_idx = 3
            else:
                col_idx = 1
            take = min(target.stop, context.shape[0])
            if target.start < take:
                return context[target.start:take, col_idx].astype(float)

        chunks = self._forecast_chunks(obs)
        if chunks.size and target.start < target.stop and schema is not None:
            col_idx = 4 if price_col == schema.price_id_eur_mwh else 5
            rel_start = max(1, target.start - int(getattr(self.env, "t", 0)))
            rel_stop = max(rel_start, target.stop - int(getattr(self.env, "t", 0)))
            start_idx = rel_start - 1
            stop_idx = min(chunks.shape[0], rel_stop - 1)
            if start_idx < stop_idx:
                return chunks[start_idx:stop_idx, col_idx].astype(float)

        episode_df = getattr(self.env, "episode_df", None)
        if episode_df is not None and schema is not None and target.start < target.stop:
            return episode_df.iloc[target][price_col].to_numpy(dtype=float)

        if chunks.size == 0:
            fallback_idx = 8 if price_col == getattr(schema, "price_id_eur_mwh", "") else 9
            return np.array([float(obs["physical"][fallback_idx])], dtype=float)
        col_idx = 4 if price_col == getattr(schema, "price_id_eur_mwh", "") else 5
        return chunks[: max(1, target.stop - target.start), col_idx].astype(float)

    def _solve_intraday_mpc(self, obs: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Compute the SIDC energy correction for the upcoming intraday block.

        This is a pure commercial decision: choose id_adjustment to balance
        SIDC trading revenue against expected imbalance exposure.

        Battery dispatch is deliberately NOT modelled here — that is the
        real-time MPC's job. The old version included p_dis/p_ch as free
        variables, which let the solver zero out imbalance for any
        id_adjustment by planning arbitrary battery dispatch. This caused
        the SIDC MPC to make overconfident commercial commitments that the
        independent real-time MPC then couldn't honour, producing large
        imbalance losses.

        Correct model: assume the battery will track the commercial schedule
        (as the real-time MPC attempts). Expected imbalance is then simply
        the gap between PV forecast and the final commercial position:

            imbalance_kw[t] = pv_forecast[t] - (dam[t] + id_adjustment)
        """
        if cp is None:
            return None
        if self._day_ahead_schedule is None:
            self._day_ahead_schedule = self._build_day_ahead_schedule(obs)

        target = self._target_slice()
        if target.start >= target.stop:
            return self._clip_intraday(np.array([0.0], dtype=np.float32))

        schema = self.env.schema
        pv = self._pv_forecast_for_slice(target, obs)
        dam = self._day_ahead_schedule[target].astype(float)
        price_id = self._slice_prices(target, obs, schema.price_id_eur_mwh)
        price_imb = self._slice_prices(target, obs, schema.price_imb_eur_mwh)
        n = min(
            len(pv),
            len(dam),
            len(price_id),
            len(price_imb),
            max(1, self.config.intraday_mpc_horizon_steps),
        )
        pv = pv[:n]
        dam = dam[:n]
        price_id = price_id[:n]
        price_imb = price_imb[:n]

        id_low = float(self.action_space["intraday"].low[0])
        id_high = float(self.action_space["intraday"].high[0])

        id_adjustment = cp.Variable()
        imbalance_abs = cp.Variable(n, nonneg=True)

        constraints = [id_adjustment >= id_low, id_adjustment <= id_high]
        revenue_terms = []
        penalty_terms = []

        for t in range(n):
            # Expected imbalance if battery perfectly tracks commercial schedule.
            expected_imbalance_kw = pv[t] - dam[t] - id_adjustment
            constraints.extend(
                [
                    imbalance_abs[t] >= expected_imbalance_kw,
                    imbalance_abs[t] >= -expected_imbalance_kw,
                ]
            )
            imbalance_weight = max(self.config.imbalance_penalty_eur_mwh, abs(float(price_imb[t])))
            revenue_terms.append(float(price_id[t]) * id_adjustment * (self.dt_hours / 1000.0))
            penalty_terms.append(imbalance_weight * imbalance_abs[t] * (self.dt_hours / 1000.0))

        problem = cp.Problem(cp.Maximize(cp.sum(revenue_terms) - cp.sum(penalty_terms)), constraints)

        try:
            problem.solve(solver=self.config.solver, warm_start=True, max_iter=self.config.solver_max_iter, verbose=False)
        except Exception:
            return None

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None
        if id_adjustment.value is None:
            return None

        return self._clip_intraday(np.array([float(id_adjustment.value)], dtype=np.float32))

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        action_mask = np.asarray(obs["action_mask"], dtype=float)
        action = self._blank_action()

        if action_mask[0] > 0.5:
            plan = self._build_d1_plan(obs)
            self._day_ahead_schedule = plan["schedule"]
            self._d1_cap_up_kw = float(plan["cap_up"])
            self._d1_cap_down_kw = float(plan["cap_down"])
            action["day_ahead"] = self._day_ahead_schedule.astype(np.float32)
            action["day_ahead_afrr"] = np.array([self._d1_cap_up_kw, self._d1_cap_down_kw], dtype=np.float32)

        if action_mask[1] > 0.5:
            intraday = self._solve_intraday_mpc(obs)
            if intraday is None:
                intraday = self._rbc_intraday_action(obs)
            self._last_intraday = intraday.astype(np.float32)
            action["intraday"] = self._last_intraday.copy()

        action["real_time"] = self._clip_low_level(self._realtime_dispatch_kw(obs))
        return action

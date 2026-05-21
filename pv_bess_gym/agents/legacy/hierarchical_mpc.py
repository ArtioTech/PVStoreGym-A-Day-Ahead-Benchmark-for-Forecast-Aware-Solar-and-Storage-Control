from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from pv_bess_gym.agents.legacy.hierarchical_baseline import HierarchicalHeuristicAgent
from pv_bess_gym.models.battery import BatteryConfig
from pv_bess_gym.models.grid import GridConfig

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - fallback path
    cp = None


@dataclass
class HierarchicalMPCConfig:
    """Tuning parameters for the hierarchical MPC baseline."""

    high_level_horizon_steps: int = 12
    low_level_horizon_steps: int = 8
    imbalance_penalty_eur_mwh: float = 140.0
    throughput_penalty_eur_mwh: float = 2.5
    bias_penalty_eur_mwh: float = 4.0
    terminal_soc_target: float = 0.55
    terminal_soc_weight: float = 12.0
    bias_limit_fraction: float = 0.25
    schedule_tracking_band_fraction: float = 0.10
    reserve_price_threshold: float = 0.0
    reserve_max_fraction: float = 0.50
    reserve_soc_buffer_fraction: float = 0.08
    reserve_sustain_hours: float = 0.50
    solver: str = "CLARABEL"
    solver_max_iter: int = 20000


class HierarchicalMPCAgent:
    """Two-layer MPC baseline aligned with the hierarchical env interface.

    The strategic layer chooses a market schedule and reserve commitments over a
    slower block horizon, while the operational layer tracks that schedule with
    battery dispatch and a small intraday correction term.
    """

    def __init__(
        self,
        action_space,
        battery_cfg: BatteryConfig,
        grid_cfg: GridConfig,
        dt_hours: float,
        decision_block_minutes: int,
        config: Optional[HierarchicalMPCConfig] = None,
    ):
        self.action_space = action_space
        self.battery_cfg = battery_cfg
        self.grid_cfg = grid_cfg
        self.dt_hours = float(dt_hours)
        self.decision_block_minutes = int(decision_block_minutes)
        self.config = config or HierarchicalMPCConfig()

        self.steps_per_block = max(1, int(round(self.decision_block_minutes / (self.dt_hours * 60.0))))
        self.eta = float(np.clip(self.battery_cfg.eta_nom, 0.85, 0.995))
        self.self_discharge_factor = 1.0 - self.battery_cfg.self_discharge_per_day * (self.dt_hours / 24.0)
        self.bias_limit_kw = self.config.bias_limit_fraction * self.battery_cfg.p_pcs_kw
        self.schedule_tracking_band_kw = self.config.schedule_tracking_band_fraction * self.battery_cfg.p_pcs_kw
        self.reserve_power_cap_kw = self.config.reserve_max_fraction * self.battery_cfg.p_pcs_kw
        self.reserve_soc_buffer_kwh = self.config.reserve_soc_buffer_fraction * self.battery_cfg.e_max_kwh
        self.fallback = HierarchicalHeuristicAgent(
            action_space=action_space,
            p_pcs_kw=self.battery_cfg.p_pcs_kw,
            reserve_fraction=0.20,
            low_price=45.0,
            high_price=95.0,
            reserve_price_threshold=max(20.0, self.config.reserve_price_threshold),
        )
        self.reset()

    def reset(self) -> None:
        self._last_high = np.zeros(4, dtype=np.float32)
        self.fallback.reset()

    def _forecast_chunks(self, forecast_vec: np.ndarray) -> np.ndarray:
        if forecast_vec.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return forecast_vec.reshape(-1, 7)

    def _clip_high_level(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space["high_level"].low,
            self.action_space["high_level"].high,
        )

    def _clip_low_level(self, action: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space["low_level"].low,
            self.action_space["low_level"].high,
        )

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

    def _project_high_level(self, action: np.ndarray, obs: dict[str, np.ndarray]) -> np.ndarray:
        high = np.asarray(action, dtype=np.float32).copy()
        reserve_up_limit_kw, reserve_down_limit_kw = self._reserve_energy_limits(float(obs["physical"][0]))

        high[2] = float(np.clip(high[2], 0.0, reserve_up_limit_kw))
        high[3] = float(np.clip(high[3], 0.0, reserve_down_limit_kw))

        total_reserve = float(high[2] + high[3])
        if total_reserve > self.reserve_power_cap_kw and total_reserve > 1e-6:
            scale = self.reserve_power_cap_kw / total_reserve
            high[2] *= scale
            high[3] *= scale
        return self._clip_high_level(high)

    def _project_low_level(self, action: np.ndarray) -> np.ndarray:
        low = np.asarray(action, dtype=np.float32).copy()
        p_batt = float(low[0])
        bias = float(np.clip(low[1], -self.bias_limit_kw, self.bias_limit_kw))

        max_discharge_kw = max(0.0, self.battery_cfg.p_pcs_kw - float(self._last_high[2]))
        max_charge_kw = max(0.0, self.battery_cfg.p_pcs_kw - float(self._last_high[3]))
        p_batt = float(np.clip(p_batt, -max_charge_kw, max_discharge_kw))
        return self._clip_low_level(np.array([p_batt, bias], dtype=np.float32))

    def _safe_low_level_fallback(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        physical = np.asarray(obs["physical"], dtype=float)
        soc = float(physical[0])
        pv_now = float(physical[2])
        pv_fcst_now = float(physical[3])
        price_dam = float(physical[7])

        p_batt = 0.0
        if price_dam <= self.fallback.low_price and soc < 0.85:
            p_batt = -0.35 * self.battery_cfg.p_pcs_kw
        elif price_dam >= self.fallback.high_price and soc > 0.20:
            p_batt = 0.35 * self.battery_cfg.p_pcs_kw

        rt_bias = np.clip(
            pv_now - pv_fcst_now,
            -self.bias_limit_kw,
            self.bias_limit_kw,
        )
        return self._project_low_level(np.array([p_batt, rt_bias], dtype=np.float32))

    def _build_horizon(self, obs: dict[str, np.ndarray], horizon_steps: int, use_actual_now: bool) -> dict[str, np.ndarray]:
        physical = np.asarray(obs["physical"], dtype=float)
        forecast_matrix = self._forecast_chunks(np.asarray(obs["forecast"], dtype=float))

        available_future = forecast_matrix.shape[0]
        resolved_horizon = max(1, min(horizon_steps, 1 + available_future))

        pv0 = float(physical[2] if use_actual_now else physical[3])
        pv = np.full(resolved_horizon, pv0, dtype=float)
        price_dam = np.full(resolved_horizon, float(physical[7]), dtype=float)
        price_id = np.full(resolved_horizon, float(physical[8]), dtype=float)
        reserve_up_price = np.full(resolved_horizon, float(physical[10]), dtype=float)
        reserve_down_price = np.full(resolved_horizon, float(physical[11]), dtype=float)
        act_up = np.zeros(resolved_horizon, dtype=float)
        act_down = np.zeros(resolved_horizon, dtype=float)
        act_up[0] = float(physical[12])
        act_down[0] = float(physical[13])

        if available_future > 0 and resolved_horizon > 1:
            take = min(available_future, resolved_horizon - 1)
            pv[1:1 + take] = forecast_matrix[:take, 0]
            price_dam[1:1 + take] = forecast_matrix[:take, 3]
            price_id[1:1 + take] = forecast_matrix[:take, 4]
            reserve_up_price[1:1 + take] = forecast_matrix[:take, 5]
            reserve_down_price[1:1 + take] = forecast_matrix[:take, 6]
            if 1 + take < resolved_horizon:
                pv[1 + take:] = pv[take]
                price_dam[1 + take:] = price_dam[take]
                price_id[1 + take:] = price_id[take]
                reserve_up_price[1 + take:] = reserve_up_price[take]
                reserve_down_price[1 + take:] = reserve_down_price[take]

        return {
            "pv_kw": pv,
            "price_dam_eur_mwh": price_dam,
            "price_id_eur_mwh": price_id,
            "reserve_up_price_eur_mw_h": reserve_up_price,
            "reserve_down_price_eur_mw_h": reserve_down_price,
            "act_up": act_up,
            "act_down": act_down,
            "current_imbalance_price_eur_mwh": np.full(resolved_horizon, float(physical[9]), dtype=float),
        }

    def _schedule_price_profile(self, price_dam: np.ndarray, price_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_steps = len(price_dam)
        num_blocks = max(1, math.ceil(num_steps / self.steps_per_block))
        schedule_price = np.zeros(num_steps, dtype=float)
        use_id_block = np.zeros(num_blocks, dtype=bool)
        for block_idx in range(num_blocks):
            sl = slice(block_idx * self.steps_per_block, min(num_steps, (block_idx + 1) * self.steps_per_block))
            dam_mean = float(np.mean(price_dam[sl]))
            id_mean = float(np.mean(price_id[sl]))
            use_id = id_mean > dam_mean
            use_id_block[block_idx] = use_id
            schedule_price[sl] = price_id[sl] if use_id else price_dam[sl]
        return schedule_price, use_id_block

    def _solve_strategic_layer(self, obs: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if cp is None:
            return None

        horizon = self._build_horizon(obs, self.config.high_level_horizon_steps, use_actual_now=False)
        pv = horizon["pv_kw"]
        price_dam = horizon["price_dam_eur_mwh"]
        price_id = horizon["price_id_eur_mwh"]
        reserve_up_price = horizon["reserve_up_price_eur_mw_h"]
        reserve_down_price = horizon["reserve_down_price_eur_mw_h"]
        act_up = horizon["act_up"]
        act_down = horizon["act_down"]
        current_imb = horizon["current_imbalance_price_eur_mwh"]
        schedule_price, use_id_block = self._schedule_price_profile(price_dam, price_id)

        num_steps = len(pv)
        num_blocks = max(1, math.ceil(num_steps / self.steps_per_block))
        schedule = cp.Variable(num_blocks)
        cap_up = cp.Variable(num_blocks, nonneg=True)
        cap_down = cp.Variable(num_blocks, nonneg=True)
        p_dis = cp.Variable(num_steps, nonneg=True)
        p_ch = cp.Variable(num_steps, nonneg=True)
        export = cp.Variable(num_steps, nonneg=True)
        grid_import = cp.Variable(num_steps, nonneg=True)
        bias = cp.Variable(num_steps)
        imbalance_abs = cp.Variable(num_steps, nonneg=True)
        bias_abs = cp.Variable(num_steps, nonneg=True)
        energy = cp.Variable(num_steps + 1)

        e_min = self.battery_cfg.soc_min * self.battery_cfg.e_max_kwh
        e_max = self.battery_cfg.soc_max * self.battery_cfg.e_max_kwh
        e_init = float(np.clip(obs["physical"][0], self.battery_cfg.soc_min, self.battery_cfg.soc_max) * self.battery_cfg.e_max_kwh)
        sched_low = float(self.action_space["high_level"].low[0])
        sched_high = float(self.action_space["high_level"].high[0])
        reserve_high = min(float(self.action_space["high_level"].high[2]), self.reserve_power_cap_kw)
        sustain_h = max(self.dt_hours, self.config.reserve_sustain_hours)

        constraints = [energy[0] == e_init, energy >= e_min, energy <= e_max]
        revenue_terms = []
        penalty_terms = []

        for t in range(num_steps):
            block_idx = t // self.steps_per_block
            act_discharge = float(act_up[t]) * cap_up[block_idx]
            act_charge = float(act_down[t]) * cap_down[block_idx]
            net_batt = p_dis[t] - p_ch[t] + act_discharge - act_charge
            scheduled_total = schedule[block_idx] + bias[t]

            constraints.extend(
                [
                    schedule[block_idx] >= sched_low,
                    schedule[block_idx] <= sched_high,
                    cap_up[block_idx] <= reserve_high,
                    cap_down[block_idx] <= reserve_high,
                    cap_up[block_idx] + cap_down[block_idx] <= self.reserve_power_cap_kw,
                    bias[t] <= self.bias_limit_kw,
                    bias[t] >= -self.bias_limit_kw,
                    p_dis[t] - p_ch[t] + cap_up[block_idx] <= self.battery_cfg.p_pcs_kw,
                    -p_dis[t] + p_ch[t] + cap_down[block_idx] <= self.battery_cfg.p_pcs_kw,
                    energy[t] >= e_min + self.reserve_soc_buffer_kwh + (sustain_h / self.eta) * cap_up[block_idx],
                    energy[t] <= e_max - self.reserve_soc_buffer_kwh - (sustain_h * self.eta) * cap_down[block_idx],
                    export[t] - grid_import[t] == pv[t] + net_batt,
                    export[t] <= self.grid_cfg.export_cap_kw,
                    grid_import[t] <= self.grid_cfg.import_cap_kw,
                    export[t] - grid_import[t] - schedule[block_idx] <= self.schedule_tracking_band_kw,
                    export[t] - grid_import[t] - schedule[block_idx] >= -self.schedule_tracking_band_kw,
                    energy[t + 1]
                    == self.self_discharge_factor * energy[t]
                    - (self.dt_hours / self.eta) * (p_dis[t] + act_discharge)
                    + (self.dt_hours * self.eta) * (p_ch[t] + act_charge),
                    imbalance_abs[t] >= export[t] - grid_import[t] - scheduled_total,
                    imbalance_abs[t] >= -(export[t] - grid_import[t] - scheduled_total),
                    bias_abs[t] >= bias[t],
                    bias_abs[t] >= -bias[t],
                ]
            )

            revenue_terms.append(schedule_price[t] * schedule[block_idx] * (self.dt_hours / 1000.0))
            revenue_terms.append(
                (
                    max(0.0, float(reserve_up_price[t]) - self.config.reserve_price_threshold) * cap_up[block_idx]
                    + max(0.0, float(reserve_down_price[t]) - self.config.reserve_price_threshold) * cap_down[block_idx]
                )
                * (self.dt_hours / 1000.0)
            )
            penalty_terms.append(
                max(self.config.imbalance_penalty_eur_mwh, abs(float(current_imb[t])))
                * imbalance_abs[t]
                * (self.dt_hours / 1000.0)
            )
            penalty_terms.append(self.config.bias_penalty_eur_mwh * bias_abs[t] * (self.dt_hours / 1000.0))
            penalty_terms.append(self.config.throughput_penalty_eur_mwh * (p_dis[t] + p_ch[t]) * (self.dt_hours / 1000.0))

        target_energy = self.config.terminal_soc_target * self.battery_cfg.e_max_kwh
        terminal_penalty = self.config.terminal_soc_weight * cp.square((energy[-1] - target_energy) / self.battery_cfg.e_max_kwh)
        objective = cp.Maximize(cp.sum(revenue_terms) - cp.sum(penalty_terms) - terminal_penalty)
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.config.solver, warm_start=True, max_iter=self.config.solver_max_iter, verbose=False)
        except Exception:
            return None
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None
        if schedule.value is None or cap_up.value is None or cap_down.value is None:
            return None

        schedule0 = float(schedule.value[0])
        cap_up0 = float(cap_up.value[0])
        cap_down0 = float(cap_down.value[0])
        if use_id_block[0]:
            high_action = np.array([0.0, schedule0, cap_up0, cap_down0], dtype=np.float32)
        else:
            high_action = np.array([schedule0, 0.0, cap_up0, cap_down0], dtype=np.float32)
        return self._project_high_level(high_action, obs)

    def _solve_operational_layer(self, obs: dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if cp is None:
            return None

        horizon = self._build_horizon(obs, self.config.low_level_horizon_steps, use_actual_now=True)
        pv = horizon["pv_kw"]
        price_id = horizon["price_id_eur_mwh"]
        current_imb = horizon["current_imbalance_price_eur_mwh"]
        act_up = horizon["act_up"]
        act_down = horizon["act_down"]

        dam_position = float(self._last_high[0])
        id_position = float(self._last_high[1])
        cap_up = float(self._last_high[2])
        cap_down = float(self._last_high[3])

        num_steps = len(pv)
        p_dis = cp.Variable(num_steps, nonneg=True)
        p_ch = cp.Variable(num_steps, nonneg=True)
        export = cp.Variable(num_steps, nonneg=True)
        grid_import = cp.Variable(num_steps, nonneg=True)
        bias = cp.Variable(num_steps)
        imbalance_abs = cp.Variable(num_steps, nonneg=True)
        bias_abs = cp.Variable(num_steps, nonneg=True)
        energy = cp.Variable(num_steps + 1)

        e_min = self.battery_cfg.soc_min * self.battery_cfg.e_max_kwh
        e_max = self.battery_cfg.soc_max * self.battery_cfg.e_max_kwh
        e_init = float(np.clip(obs["physical"][0], self.battery_cfg.soc_min, self.battery_cfg.soc_max) * self.battery_cfg.e_max_kwh)

        constraints = [energy[0] == e_init, energy >= e_min, energy <= e_max]
        penalty_terms = []

        for t in range(num_steps):
            act_discharge = float(act_up[t]) * cap_up
            act_charge = float(act_down[t]) * cap_down
            net_batt = p_dis[t] - p_ch[t] + act_discharge - act_charge
            scheduled_total = dam_position + id_position + bias[t]
            imbalance_price_weight = max(self.config.imbalance_penalty_eur_mwh, abs(float(current_imb[t])))

            constraints.extend(
                [
                    bias[t] <= self.bias_limit_kw,
                    bias[t] >= -self.bias_limit_kw,
                    p_dis[t] - p_ch[t] + cap_up <= self.battery_cfg.p_pcs_kw,
                    -p_dis[t] + p_ch[t] + cap_down <= self.battery_cfg.p_pcs_kw,
                    export[t] - grid_import[t] == pv[t] + net_batt,
                    export[t] <= self.grid_cfg.export_cap_kw,
                    grid_import[t] <= self.grid_cfg.import_cap_kw,
                    energy[t + 1]
                    == self.self_discharge_factor * energy[t]
                    - (self.dt_hours / self.eta) * (p_dis[t] + act_discharge)
                    + (self.dt_hours * self.eta) * (p_ch[t] + act_charge),
                    imbalance_abs[t] >= export[t] - grid_import[t] - scheduled_total,
                    imbalance_abs[t] >= -(export[t] - grid_import[t] - scheduled_total),
                    bias_abs[t] >= bias[t],
                    bias_abs[t] >= -bias[t],
                ]
            )

            penalty_terms.append(imbalance_price_weight * imbalance_abs[t] * (self.dt_hours / 1000.0))
            penalty_terms.append(self.config.bias_penalty_eur_mwh * bias_abs[t] * (self.dt_hours / 1000.0))
            penalty_terms.append(self.config.throughput_penalty_eur_mwh * (p_dis[t] + p_ch[t]) * (self.dt_hours / 1000.0))

        target_energy = self.config.terminal_soc_target * self.battery_cfg.e_max_kwh
        terminal_penalty = self.config.terminal_soc_weight * cp.square((energy[-1] - target_energy) / self.battery_cfg.e_max_kwh)
        objective = cp.Maximize(-cp.sum(penalty_terms) - terminal_penalty)
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=self.config.solver, warm_start=True, max_iter=self.config.solver_max_iter, verbose=False)
        except Exception:
            return None
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return None
        if p_dis.value is None or p_ch.value is None or bias.value is None:
            return None

        p_batt0 = float(p_dis.value[0] - p_ch.value[0])
        bias0 = float(bias.value[0])
        return self._project_low_level(np.array([p_batt0, bias0], dtype=np.float32))

    def act(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        action_mask = np.asarray(obs["action_mask"], dtype=float)
        if action_mask[0] > 0.5:
            strategic = self._solve_strategic_layer(obs)
            if strategic is None:
                fallback_action = self.fallback.act(obs)
                self._last_high = self._project_high_level(np.asarray(fallback_action["high_level"], dtype=np.float32), obs)
            else:
                self._last_high = strategic.astype(np.float32)
            self.fallback._last_high = self._last_high.copy()

        low_action = self._solve_operational_layer(obs)
        if low_action is None:
            low_action = self._safe_low_level_fallback(obs)

        return {
            "high_level": self._clip_high_level(self._last_high),
            "low_level": self._project_low_level(low_action),
        }

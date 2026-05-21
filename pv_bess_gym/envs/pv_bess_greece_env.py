from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from pv_bess_gym.data.signals import SignalSchema
from pv_bess_gym.models.battery import Battery, BatteryConfig
from pv_bess_gym.models.grid import Grid, GridConfig
from pv_bess_gym.models.pv import PVConfig, PVModel
from pv_bess_gym.models.settlement import (
    SettlementBreakdown,
    eur_from_kwh,
    kwh_from_kw,
    reserve_capacity_eur,
    settle_energy_layers,
)


@dataclass
class PlantConfig:
    """Physical plant definition.

    The plant is split into three components:
    - PV field and inverter
    - Battery and PCS
    - Grid connection point
    """

    pv: PVConfig
    battery: BatteryConfig
    grid: GridConfig


@dataclass
class MarketConfig:
    """Simulation-time and settlement parameters."""

    dt_h: float = 0.25
    afrr_block_minutes: int = 60
    violation_penalty_eur_per_kw: float = 1000.0
    fallback_shortfall_penalty_eur_per_mwh: float = 200.0
    fallback_unavailable_penalty_eur_per_mw: float = 0.0


@dataclass
class HierarchicalConfig:
    """Controls the slower strategic decision layer."""

    decision_block_minutes: int = 60
    expose_action_mask: bool = True


@dataclass
class DayAheadMarketTiming:
    """Timing abstraction for the unified Greek-market environment.

    The real market sequence is D-1 day-ahead commitment, intraday corrections
    closer to delivery, and real-time physical dispatch with imbalance
    settlement. This config keeps that sequence explicit while remaining simple
    enough for Gym benchmarking:
    - the full DAM schedule is committed once before the delivery episode;
    - intraday corrections are additive and respect a configurable lead time;
    - the low-level action only dispatches the battery, it does not create an
      artificial real-time trade.
    """

    intraday_block_minutes: int = 60
    intraday_gate_closure_minutes: int = 60
    expose_action_mask: bool = True


class _BasePVBESSGreeceEnv(gym.Env):
    """Internal simulator core shared by higher-level environment interfaces.

    The simulator uses exogenous weather and market signals, a PV model, a
    battery model, a grid model, and a simplified commercial settlement logic.
    It is designed for algorithm benchmarking rather than market-compliant
    operation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        signals: pd.DataFrame,
        schema: SignalSchema,
        plant: PlantConfig,
        market: MarketConfig,
        episode_len: int = 96,
        soc_init: float = 0.10,
        include_forecasts: bool = True,
        forecast_horizon_steps: int = 8,
    ):
        super().__init__()
        self.df_all = signals.reset_index(drop=True)
        self.schema = schema
        self.plant = plant
        self.market = market
        self.episode_len = int(episode_len)
        self.soc_init = float(soc_init)
        self.include_forecasts = bool(include_forecasts)
        self.forecast_horizon_steps = int(forecast_horizon_steps)

        self.pv_model = PVModel(plant.pv)
        self.grid = Grid(plant.grid)
        self.battery: Optional[Battery] = None

        self.afrr_block_steps = max(1, int(round(market.afrr_block_minutes / (market.dt_h * 60))))
        self.cap_up_commit_kw = 0.0
        self.cap_down_commit_kw = 0.0
        self.block_steps_left = 0

        self.episode_df: Optional[pd.DataFrame] = None
        self.t = 0
        self.cum_profit = 0.0
        self.log: List[Dict[str, Any]] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}
        start_idx = int(options.get("start_idx", 0))
        if start_idx + self.episode_len > len(self.df_all):
            start_idx = max(0, len(self.df_all) - self.episode_len)

        self.episode_df = self.df_all.iloc[start_idx:start_idx + self.episode_len].reset_index(drop=True)
        self.t = 0
        self.cum_profit = 0.0
        self.log = []
        self.battery = Battery(self.plant.battery, soc_init=float(options.get("soc_init", self.soc_init)))
        self.cap_up_commit_kw = 0.0
        self.cap_down_commit_kw = 0.0
        self.block_steps_left = 0
        return {"start_idx": start_idx}

    def _row(self, t: int) -> pd.Series:
        assert self.episode_df is not None
        return self.episode_df.iloc[t]

    def _time_features(self, ts: pd.Timestamp) -> tuple[float, float]:
        ts = pd.to_datetime(ts)
        minutes = ts.hour * 60 + ts.minute
        angle = 2 * np.pi * (minutes / (24 * 60))
        return float(np.sin(angle)), float(np.cos(angle))

    def _weather_power(self, row: pd.Series, forecast: bool = False) -> float:
        if forecast:
            ghi_col = self.schema.ghi_fcst_wm2 if self.schema.ghi_fcst_wm2 in row.index else self.schema.ghi_wm2
            temp_col = self.schema.temp_air_fcst_c if self.schema.temp_air_fcst_c in row.index else self.schema.temp_air_c
            wind_col = self.schema.wind_fcst_mps if (self.schema.wind_fcst_mps and self.schema.wind_fcst_mps in row.index) else self.schema.wind_mps
        else:
            ghi_col = self.schema.ghi_wm2
            temp_col = self.schema.temp_air_c
            wind_col = self.schema.wind_mps

        wind = None if (wind_col is None or wind_col not in row.index) else float(row[wind_col])
        return self.pv_model.ac_power_kw(
            pd.to_datetime(row[self.schema.ts_col]),
            float(row[ghi_col]),
            float(row[temp_col]),
            wind,
        )

    def _activation_kw(self, row: pd.Series, col: Optional[str], cap_kw: float) -> float:
        if col is None or col not in row.index:
            return 0.0
        value = float(row[col])
        if 0.0 <= value <= 1.0:
            return value * cap_kw
        return min(abs(value), cap_kw)

    def _latch_afrr_block(self, cap_up_kw: float, cap_down_kw: float) -> None:
        if self.block_steps_left <= 0:
            self.cap_up_commit_kw = float(cap_up_kw)
            self.cap_down_commit_kw = float(cap_down_kw)
            self.block_steps_left = self.afrr_block_steps

    def _settle_step(
        self,
        row: pd.Series,
        dam_position_kw: float,
        id_adjustment_kw: float,
        p_batt_base_kw: float,
        cap_up_req_kw: float,
        cap_down_req_kw: float,
    ) -> tuple[float, Dict[str, Any]]:
        assert self.battery is not None
        dt = self.market.dt_h

        self._latch_afrr_block(cap_up_req_kw, cap_down_req_kw)
        cap_up_kw = self.cap_up_commit_kw
        cap_down_kw = self.cap_down_commit_kw

        pv_kw = self._weather_power(row, forecast=False)
        pv_fcst_kw = self._weather_power(row, forecast=True)

        ppcs = self.plant.battery.p_pcs_kw
        violation_kw = 0.0
        if p_batt_base_kw + cap_up_kw > ppcs + 1e-6:
            violation_kw += p_batt_base_kw + cap_up_kw - ppcs
        if (-p_batt_base_kw) + cap_down_kw > ppcs + 1e-6:
            violation_kw += (-p_batt_base_kw) + cap_down_kw - ppcs

        act_up_kw = self._activation_kw(row, self.schema.afrr_act_up, cap_up_kw)
        act_down_kw = self._activation_kw(row, self.schema.afrr_act_down, cap_down_kw)
        p_act_kw = act_up_kw - act_down_kw

        p_batt_req_kw = p_batt_base_kw + p_act_kw
        batt_res = self.battery.step(p_batt_req_kw, dt_h=dt)
        p_batt_real_kw = float(batt_res["p_real_kw"])

        shortfall_kw = max(0.0, abs(p_batt_req_kw) - abs(p_batt_real_kw))
        shortfall_mwh = (shortfall_kw * dt) / 1000.0

        net_injection_kw = pv_kw + p_batt_real_kw
        grid_res = self.grid.apply(net_injection_kw)
        export_kw = float(grid_res["export_kw"])
        import_kw = float(grid_res["import_kw"])

        price_dam = float(row[self.schema.price_dam_eur_mwh])
        price_id = float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else price_dam
        price_imb = float(row[self.schema.price_imb_eur_mwh])

        energy_layers = settle_energy_layers(
            dam_position_kw=dam_position_kw,
            id_adjustment_kw=id_adjustment_kw,
            actual_export_kw=export_kw,
            actual_import_kw=import_kw,
            price_dam_eur_mwh=price_dam,
            price_id_eur_mwh=price_id,
            price_imb_eur_mwh=price_imb,
            dt_h=dt,
        )

        afrr_cap_eur = (
            reserve_capacity_eur(cap_up_kw, float(row[self.schema.afrr_cap_up_eur_mw_h]), dt)
            + reserve_capacity_eur(cap_down_kw, float(row[self.schema.afrr_cap_down_eur_mw_h]), dt)
        )

        en_up = price_imb if (self.schema.afrr_en_up_eur_mwh is None or self.schema.afrr_en_up_eur_mwh not in row.index) else float(row[self.schema.afrr_en_up_eur_mwh])
        en_down = price_imb if (self.schema.afrr_en_down_eur_mwh is None or self.schema.afrr_en_down_eur_mwh not in row.index) else float(row[self.schema.afrr_en_down_eur_mwh])
        afrr_activation_eur = (
            eur_from_kwh(kwh_from_kw(act_up_kw, dt), en_up)
            + eur_from_kwh(kwh_from_kw(act_down_kw, dt), en_down)
        )

        shortfall_penalty = (
            float(row[self.schema.penalty_shortfall_eur_mwh])
            if (self.schema.penalty_shortfall_eur_mwh and self.schema.penalty_shortfall_eur_mwh in row.index)
            else self.market.fallback_shortfall_penalty_eur_per_mwh
        )
        unavailable_penalty = (
            float(row[self.schema.penalty_unavailable_eur_mw])
            if (self.schema.penalty_unavailable_eur_mw and self.schema.penalty_unavailable_eur_mw in row.index)
            else self.market.fallback_unavailable_penalty_eur_per_mw
        )

        penalties_eur = (
            self.market.violation_penalty_eur_per_kw * violation_kw
            + shortfall_penalty * shortfall_mwh
            + unavailable_penalty * (violation_kw / 1000.0)
        )

        breakdown = SettlementBreakdown(
            dam_eur=energy_layers["dam_eur"],
            id_eur=energy_layers["id_eur"],
            imbalance_eur=energy_layers["imbalance_eur"],
            afrr_capacity_eur=afrr_cap_eur,
            afrr_activation_eur=afrr_activation_eur,
            degradation_eur=float(batt_res["deg_cost"]),
            penalties_eur=penalties_eur,
        )

        reward = breakdown.total_eur
        self.cum_profit += reward

        info = {
            "t": self.t,
            "timestamp": pd.to_datetime(row[self.schema.ts_col]).isoformat(),
            "reward": float(reward),
            "cum_profit": float(self.cum_profit),
            "pv_kw": float(pv_kw),
            "pv_fcst_kw": float(pv_fcst_kw),
            "dam_position_kw": float(dam_position_kw),
            "id_adjustment_kw": float(id_adjustment_kw),
            "scheduled_position_kw": float(dam_position_kw + id_adjustment_kw),
            "p_batt_base_kw": float(p_batt_base_kw),
            "p_batt_req_kw": float(p_batt_req_kw),
            "p_batt_real_kw": float(p_batt_real_kw),
            "soc": float(self.battery.soc),
            "export_kw": export_kw,
            "import_kw": import_kw,
            "curtailed_kw": float(grid_res["curtailed_kw"]),
            "cap_up_commit_kw": float(cap_up_kw),
            "cap_down_commit_kw": float(cap_down_kw),
            "act_up_kw": float(act_up_kw),
            "act_down_kw": float(act_down_kw),
            "dam_eur": float(breakdown.dam_eur),
            "id_eur": float(breakdown.id_eur),
            "imbalance_eur": float(breakdown.imbalance_eur),
            "afrr_capacity_eur": float(breakdown.afrr_capacity_eur),
            "afrr_activation_eur": float(breakdown.afrr_activation_eur),
            "degradation_eur": float(breakdown.degradation_eur),
            "penalties_eur": float(breakdown.penalties_eur),
            "violation_kw": float(violation_kw),
            "shortfall_mwh": float(shortfall_mwh),
            "penalty_params": {
                "violation_penalty_eur_per_kw": self.market.violation_penalty_eur_per_kw,
                "shortfall_penalty_eur_per_mwh": float(shortfall_penalty),
                "unavailable_penalty_eur_per_mw": float(unavailable_penalty),
            },
        }
        return reward, info


class HierarchicalPVBESSGreeceEnv(_BasePVBESSGreeceEnv):
    """Hierarchical Gymnasium environment for PV+BESS market participation.

    The environment separates strategic and operational decisions:

    High-level action, applied only at decision boundaries:
    - day-ahead position at the point of common coupling
    - intraday adjustment
    - upward aFRR capacity commitment
    - downward aFRR capacity commitment

    Low-level action, applied at every step:
    - battery dispatch setpoint
    - real-time commercial bias, used here as a fast intraday correction term

    Observation dictionary:
    - ``physical``: current plant state, current prices, current activation signal,
      and timing features
    - ``forecast``: future forecast features over a finite horizon
    - ``action_mask``: tells the agent when a new high-level decision is due
    """

    def __init__(
        self,
        signals: pd.DataFrame,
        schema: SignalSchema,
        plant: PlantConfig,
        market: MarketConfig,
        episode_len: int = 96,
        soc_init: float = 0.10,
        include_forecasts: bool = True,
        forecast_horizon_steps: int = 8,
        hierarchy: Optional[HierarchicalConfig] = None,
    ):
        self.hierarchy = hierarchy or HierarchicalConfig()
        self.hl_dam_position_kw = 0.0
        self.hl_id_adjustment_kw = 0.0
        self.hl_cap_up_kw = 0.0
        self.hl_cap_down_kw = 0.0
        self.hl_steps_left = 0
        super().__init__(
            signals=signals,
            schema=schema,
            plant=plant,
            market=market,
            episode_len=episode_len,
            soc_init=soc_init,
            include_forecasts=include_forecasts,
            forecast_horizon_steps=forecast_horizon_steps,
        )
        self._build_spaces()

    def _build_spaces(self) -> None:
        ppcs = self.plant.battery.p_pcs_kw
        max_sched = max(self.plant.grid.export_cap_kw, self.plant.grid.import_cap_kw)
        self.decision_block_steps = max(1, int(round(self.hierarchy.decision_block_minutes / (self.market.dt_h * 60))))

        self.action_space = spaces.Dict(
            {
                "high_level": spaces.Box(
                    low=np.array([-max_sched, -max_sched, 0.0, 0.0], dtype=np.float32),
                    high=np.array([max_sched, max_sched, ppcs, ppcs], dtype=np.float32),
                    dtype=np.float32,
                ),
                "low_level": spaces.Box(
                    low=np.array([-ppcs, -max_sched], dtype=np.float32),
                    high=np.array([ppcs, max_sched], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        physical_dim = 18
        forecast_dim = self.forecast_horizon_steps * 7 if self.include_forecasts else 0
        self.observation_space = spaces.Dict(
            {
                "physical": spaces.Box(
                    low=np.full(physical_dim, -1e9, dtype=np.float32),
                    high=np.full(physical_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "forecast": spaces.Box(
                    low=np.full(forecast_dim, -1e9, dtype=np.float32),
                    high=np.full(forecast_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.hl_dam_position_kw = 0.0
        self.hl_id_adjustment_kw = 0.0
        self.hl_cap_up_kw = 0.0
        self.hl_cap_down_kw = 0.0
        self.hl_steps_left = 0
        info = super().reset(seed=seed, options=options)
        return self._get_obs(), info

    def _strategic_due(self) -> bool:
        return getattr(self, "hl_steps_left", 0) <= 0

    def _forecast_matrix(self) -> List[float]:
        out: List[float] = []
        if not self.include_forecasts:
            return out
        for k in range(1, self.forecast_horizon_steps + 1):
            tt = min(self.t + k, self.episode_len - 1)
            row = self._row(tt)
            out.extend(
                [
                    self._weather_power(row, forecast=True),
                    float(row[self.schema.ghi_fcst_wm2]) if self.schema.ghi_fcst_wm2 in row.index else float(row[self.schema.ghi_wm2]),
                    float(row[self.schema.temp_air_fcst_c]) if self.schema.temp_air_fcst_c in row.index else float(row[self.schema.temp_air_c]),
                    float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.afrr_cap_up_eur_mw_h]),
                    float(row[self.schema.afrr_cap_down_eur_mw_h]),
                ]
            )
        return out

    def _get_obs(self) -> Dict[str, np.ndarray]:
        row = self._row(self.t)
        tod_sin, tod_cos = self._time_features(pd.to_datetime(row[self.schema.ts_col]))
        wind = 0.0 if (self.schema.wind_mps is None or self.schema.wind_mps not in row.index) else float(row[self.schema.wind_mps])
        act_up_signal = float(row[self.schema.afrr_act_up]) if (self.schema.afrr_act_up and self.schema.afrr_act_up in row.index) else 0.0
        act_down_signal = float(row[self.schema.afrr_act_down]) if (self.schema.afrr_act_down and self.schema.afrr_act_down in row.index) else 0.0

        physical = np.array(
            [
                float(self.battery.soc),
                float(self.battery.e_kwh),
                float(self._weather_power(row, forecast=False)),
                float(self._weather_power(row, forecast=True)),
                float(row[self.schema.ghi_wm2]),
                float(row[self.schema.temp_air_c]),
                float(wind),
                float(row[self.schema.price_dam_eur_mwh]),
                float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                float(row[self.schema.price_imb_eur_mwh]),
                float(row[self.schema.afrr_cap_up_eur_mw_h]),
                float(row[self.schema.afrr_cap_down_eur_mw_h]),
                float(act_up_signal),
                float(act_down_signal),
                float(getattr(self, "hl_steps_left", 0)),
                float(self.block_steps_left),
                tod_sin,
                tod_cos,
            ],
            dtype=np.float32,
        )

        return {
            "physical": physical,
            "forecast": np.array(self._forecast_matrix(), dtype=np.float32),
            "action_mask": np.array([1.0 if self._strategic_due() else 0.0, 1.0], dtype=np.float32),
        }

    def _maybe_update_high_level(self, high_level: np.ndarray) -> bool:
        if self._strategic_due():
            hl = np.array(high_level, dtype=np.float32)
            self.hl_dam_position_kw = float(hl[0])
            self.hl_id_adjustment_kw = float(hl[1])
            self.hl_cap_up_kw = float(max(0.0, hl[2]))
            self.hl_cap_down_kw = float(max(0.0, hl[3]))
            self.hl_steps_left = self.decision_block_steps
            return True
        return False

    def step(self, action: Dict[str, np.ndarray]):
        row = self._row(self.t)
        high_level = action.get("high_level", np.zeros(4, dtype=np.float32))
        low_level = np.array(action.get("low_level", np.zeros(2, dtype=np.float32)), dtype=np.float32)
        high_level_applied = self._maybe_update_high_level(high_level)

        p_batt_base_kw = float(low_level[0])
        realtime_trading_bias_kw = float(low_level[1])
        reward, info = self._settle_step(
            row=row,
            dam_position_kw=self.hl_dam_position_kw,
            id_adjustment_kw=self.hl_id_adjustment_kw + realtime_trading_bias_kw,
            p_batt_base_kw=p_batt_base_kw,
            cap_up_req_kw=self.hl_cap_up_kw,
            cap_down_req_kw=self.hl_cap_down_kw,
        )
        info.update(
            {
                "high_level_due": bool(self._strategic_due()),
                "high_level_applied": bool(high_level_applied),
                "hl_dam_position_kw": float(self.hl_dam_position_kw),
                "hl_id_adjustment_kw": float(self.hl_id_adjustment_kw),
                "hl_cap_up_kw": float(self.hl_cap_up_kw),
                "hl_cap_down_kw": float(self.hl_cap_down_kw),
                "realtime_trading_bias_kw": float(realtime_trading_bias_kw),
            }
        )
        self.log.append(info)

        self.block_steps_left = max(0, self.block_steps_left - 1)
        self.hl_steps_left = max(0, self.hl_steps_left - 1)
        self.t += 1
        terminated = self.t >= self.episode_len
        if terminated:
            next_obs = {
                "physical": np.zeros(self.observation_space["physical"].shape, dtype=np.float32),
                "forecast": np.zeros(self.observation_space["forecast"].shape, dtype=np.float32),
                "action_mask": np.zeros(self.observation_space["action_mask"].shape, dtype=np.float32),
            }
        else:
            next_obs = self._get_obs()
        return next_obs, float(reward), terminated, False, info


class DayAheadPVBESSGreeceEnv(_BasePVBESSGreeceEnv):
    """Greek-market-sequenced Gymnasium environment for PV+BESS operation.

    This class is kept as a backwards-compatible name. The canonical public
    class is :class:`UnifiedPVBESSGreeceEnv`, which inherits this implementation.
    Both follow the cleaner market chronology needed for a day-ahead
    formulation:

    1. Day-ahead layer: a full delivery-episode DAM schedule is committed once.
       In market terms this represents a D-1 schedule for delivery day D. The
       aFRR capacity offer is also committed at D-1 simultaneously.
    2. Intraday layer: corrections are additive to the DAM schedule and are
       written only into future periods after the configured gate closure.
       Energy correction only; aFRR capacity is fixed from D-1.
    3. Real-time layer: the action dispatches the battery only. Deviations from
       DAM + IDM are settled through imbalance prices rather than hidden inside
       a real-time trading action.

    Action dictionary:
    - ``day_ahead``: vector of length ``episode_len`` with the fixed DAM
      schedule at the PCC in kW. Required only when no schedule was supplied at
      reset and the day-ahead mask is active.
    - ``day_ahead_afrr``: ``[cap_up_kw, cap_down_kw]`` aFRR capacity offer
      committed at D-1 alongside the DAM schedule, fixed for the full delivery day.
    - ``intraday``: ``[id_adjustment_kw]`` additive SIDC energy correction only.
      The energy correction is written to a future block after the intraday gate
      closure.
    - ``real_time``: ``[p_batt_base_kw]`` real-time battery setpoint. The older
      key ``low_level`` is accepted as an input alias for compatibility.

    Observation dictionary:
    - ``physical``: current plant/market state, including d1_cap_up_kw and d1_cap_down_kw
      at the end.
    - ``forecast``: rolling short-horizon context for intraday and real-time
      operation.
    - ``day_ahead_context``: full delivery-episode context for the DAM schedule.
      Each row is ``[pv_fcst_kw, price_dam, price_id, price_imb,
      afrr_cap_up, afrr_cap_down, tod_sin, tod_cos]`` and the matrix is
      flattened row-major.
    - ``action_mask``: ``[day_ahead_due, intraday_due, real_time_due]``.
    """

    market_mode = "day_ahead"

    def __init__(
        self,
        signals: pd.DataFrame,
        schema: SignalSchema,
        plant: PlantConfig,
        market: MarketConfig,
        episode_len: int = 96,
        soc_init: float = 0.10,
        include_forecasts: bool = True,
        forecast_horizon_steps: int = 8,
        timing: Optional[DayAheadMarketTiming] = None,
    ):
        self.timing = timing or DayAheadMarketTiming()
        self.day_ahead_schedule_kw = np.zeros(int(episode_len), dtype=float)
        self.id_adjustment_profile_kw = np.zeros(int(episode_len), dtype=float)
        self.day_ahead_committed = False
        self.day_ahead_source = "uncommitted"
        self.id_steps_left = 0
        self.current_id_offer_kw = 0.0
        self.d1_cap_up_kw = 0.0
        self.d1_cap_down_kw = 0.0
        self.last_intraday_target_start = 0
        self.last_intraday_target_stop = 0
        super().__init__(
            signals=signals,
            schema=schema,
            plant=plant,
            market=market,
            episode_len=episode_len,
            soc_init=soc_init,
            include_forecasts=include_forecasts,
            forecast_horizon_steps=forecast_horizon_steps,
        )
        self._build_spaces()

    def _build_spaces(self) -> None:
        ppcs = self.plant.battery.p_pcs_kw
        max_sched = max(self.plant.grid.export_cap_kw, self.plant.grid.import_cap_kw)
        self.intraday_block_steps = max(
            1,
            int(round(self.timing.intraday_block_minutes / (self.market.dt_h * 60))),
        )
        self.intraday_gate_steps = max(
            0,
            int(round(self.timing.intraday_gate_closure_minutes / (self.market.dt_h * 60))),
        )

        self.action_space = spaces.Dict(
            {
                "day_ahead": spaces.Box(
                    low=np.full(self.episode_len, -max_sched, dtype=np.float32),
                    high=np.full(self.episode_len, max_sched, dtype=np.float32),
                    dtype=np.float32,
                ),
                "day_ahead_afrr": spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([ppcs, ppcs], dtype=np.float32),
                    dtype=np.float32,
                ),
                "intraday": spaces.Box(
                    low=np.array([-max_sched], dtype=np.float32),
                    high=np.array([max_sched], dtype=np.float32),
                    dtype=np.float32,
                ),
                "real_time": spaces.Box(
                    low=np.array([-ppcs], dtype=np.float32),
                    high=np.array([ppcs], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

        physical_dim = 24
        forecast_dim = self.forecast_horizon_steps * 9 if self.include_forecasts else 0
        self.observation_space = spaces.Dict(
            {
                "physical": spaces.Box(
                    low=np.full(physical_dim, -1e9, dtype=np.float32),
                    high=np.full(physical_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "forecast": spaces.Box(
                    low=np.full(forecast_dim, -1e9, dtype=np.float32),
                    high=np.full(forecast_dim, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "day_ahead_context": spaces.Box(
                    low=np.full(self.episode_len * 8, -1e9, dtype=np.float32),
                    high=np.full(self.episode_len * 8, 1e9, dtype=np.float32),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32),
            }
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        options = options or {}
        info = super().reset(seed=seed, options=options)

        self.day_ahead_schedule_kw = np.zeros(self.episode_len, dtype=float)
        self.id_adjustment_profile_kw = np.zeros(self.episode_len, dtype=float)
        self.day_ahead_committed = False
        self.day_ahead_source = "uncommitted"
        self.id_steps_left = 0
        self.current_id_offer_kw = 0.0
        self.d1_cap_up_kw = 0.0
        self.d1_cap_down_kw = 0.0
        self.last_intraday_target_start = 0
        self.last_intraday_target_stop = 0

        if "day_ahead_schedule_kw" in options:
            self._commit_day_ahead_schedule(options["day_ahead_schedule_kw"], source="reset_options")
        if "id_adjustment_profile_kw" in options:
            self.id_adjustment_profile_kw = self._coerce_profile(
                options["id_adjustment_profile_kw"],
                name="id_adjustment_profile_kw",
            )
        if "d1_cap_up_kw" in options:
            self.d1_cap_up_kw = float(options["d1_cap_up_kw"])
        if "d1_cap_down_kw" in options:
            self.d1_cap_down_kw = float(options["d1_cap_down_kw"])

        info.update(
            {
                "market_mode": self.market_mode,
                "day_ahead_committed": bool(self.day_ahead_committed),
                "day_ahead_source": self.day_ahead_source,
                "intraday_gate_closure_steps": int(self.intraday_gate_steps),
                "intraday_gate_closure_minutes": int(self.timing.intraday_gate_closure_minutes),
            }
        )
        return self._get_obs(), info

    def _coerce_profile(self, values: Any, *, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size != self.episode_len:
            raise ValueError(
                f"{name} must contain exactly {self.episode_len} values for this delivery episode; "
                f"got {arr.size}."
            )
        max_sched = max(self.plant.grid.export_cap_kw, self.plant.grid.import_cap_kw)
        return np.clip(arr.astype(float), -max_sched, max_sched)

    def _commit_day_ahead_schedule(self, schedule_kw: Any, *, source: str) -> None:
        self.day_ahead_schedule_kw = self._coerce_profile(schedule_kw, name="day_ahead_schedule_kw")
        self.day_ahead_committed = True
        self.day_ahead_source = source

    def _day_ahead_due(self) -> bool:
        return self.t == 0 and not self.day_ahead_committed

    def _intraday_due(self) -> bool:
        return self.id_steps_left <= 0

    def _intraday_target_slice(self) -> slice:
        start = min(self.episode_len, self.t + self.intraday_gate_steps)
        stop = min(self.episode_len, start + self.intraday_block_steps)
        return slice(start, stop)

    def _maybe_commit_day_ahead_from_action(self, action: Dict[str, Any]) -> bool:
        if not self._day_ahead_due():
            return False
        if "day_ahead" not in action:
            raise ValueError(
                "Day-ahead schedule is due but action['day_ahead'] is missing. "
                "Provide a full episode-length DAM schedule, or pass "
                "reset(options={'day_ahead_schedule_kw': ...}) before stepping."
            )
        self._commit_day_ahead_schedule(action["day_ahead"], source="action")

        # Read D-1 aFRR capacity offer if present
        if "day_ahead_afrr" in action:
            afrr = np.asarray(action["day_ahead_afrr"], dtype=np.float32).reshape(-1)
            if afrr.size >= 2:
                ppcs = self.plant.battery.p_pcs_kw
                self.d1_cap_up_kw = float(np.clip(afrr[0], 0.0, ppcs))
                self.d1_cap_down_kw = float(np.clip(afrr[1], 0.0, ppcs))
        return True

    def _maybe_update_intraday(self, intraday: Any) -> bool:
        if not self._intraday_due():
            return False

        raw = np.zeros(1, dtype=np.float32) if intraday is None else np.asarray(intraday, dtype=np.float32).reshape(-1)
        if raw.size != 1:
            raise ValueError(
                "action['intraday'] must contain [id_adjustment_kw] only. "
                "aFRR capacity is committed D-1 via action['day_ahead_afrr']."
            )

        clipped = np.clip(raw, self.action_space["intraday"].low, self.action_space["intraday"].high)
        id_adjustment_kw = float(clipped[0])

        target = self._intraday_target_slice()
        self.last_intraday_target_start = int(target.start)
        self.last_intraday_target_stop = int(target.stop)
        if target.start < target.stop:
            self.id_adjustment_profile_kw[target] = id_adjustment_kw

        self.current_id_offer_kw = id_adjustment_kw
        self.id_steps_left = self.intraday_block_steps
        return True

    def _forecast_matrix(self) -> List[float]:
        out: List[float] = []
        if not self.include_forecasts:
            return out

        for k in range(1, self.forecast_horizon_steps + 1):
            tt = min(self.t + k, self.episode_len - 1)
            row = self._row(tt)
            day_ahead_kw = float(self.day_ahead_schedule_kw[tt])
            id_adjustment_kw = float(self.id_adjustment_profile_kw[tt])
            out.extend(
                [
                    self._weather_power(row, forecast=True),
                    float(row[self.schema.ghi_fcst_wm2]) if self.schema.ghi_fcst_wm2 in row.index else float(row[self.schema.ghi_wm2]),
                    float(row[self.schema.temp_air_fcst_c]) if self.schema.temp_air_fcst_c in row.index else float(row[self.schema.temp_air_c]),
                    float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.price_imb_eur_mwh]),
                    float(row[self.schema.afrr_cap_up_eur_mw_h]),
                    float(row[self.schema.afrr_cap_down_eur_mw_h]),
                    day_ahead_kw + id_adjustment_kw,
                ]
            )
        return out

    def _day_ahead_context_matrix(self) -> List[float]:
        """Expose full-day scheduling data through the Gym observation.

        This keeps the upper-level optimizer/RL policy on the normal Gym
        contract: it can compute the delivery-day DAM schedule from observation
        data instead of reading private environment internals.
        """
        out: List[float] = []
        for tt in range(self.episode_len):
            row = self._row(tt)
            tod_sin, tod_cos = self._time_features(pd.to_datetime(row[self.schema.ts_col]))
            out.extend(
                [
                    float(self._weather_power(row, forecast=True)),
                    float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                    float(row[self.schema.price_imb_eur_mwh]),
                    float(row[self.schema.afrr_cap_up_eur_mw_h]),
                    float(row[self.schema.afrr_cap_down_eur_mw_h]),
                    tod_sin,
                    tod_cos,
                ]
            )
        return out

    def _get_obs(self) -> Dict[str, np.ndarray]:
        row = self._row(self.t)
        tod_sin, tod_cos = self._time_features(pd.to_datetime(row[self.schema.ts_col]))
        wind = 0.0 if (self.schema.wind_mps is None or self.schema.wind_mps not in row.index) else float(row[self.schema.wind_mps])
        act_up_signal = float(row[self.schema.afrr_act_up]) if (self.schema.afrr_act_up and self.schema.afrr_act_up in row.index) else 0.0
        act_down_signal = float(row[self.schema.afrr_act_down]) if (self.schema.afrr_act_down and self.schema.afrr_act_down in row.index) else 0.0
        day_ahead_kw = float(self.day_ahead_schedule_kw[self.t])
        id_adjustment_kw = float(self.id_adjustment_profile_kw[self.t])

        physical = np.array(
            [
                float(self.battery.soc),
                float(self.battery.e_kwh),
                float(self._weather_power(row, forecast=False)),
                float(self._weather_power(row, forecast=True)),
                float(row[self.schema.ghi_wm2]),
                float(row[self.schema.temp_air_c]),
                float(wind),
                float(row[self.schema.price_dam_eur_mwh]),
                float(row[self.schema.price_id_eur_mwh]) if self.schema.price_id_eur_mwh in row.index else float(row[self.schema.price_dam_eur_mwh]),
                float(row[self.schema.price_imb_eur_mwh]),
                float(row[self.schema.afrr_cap_up_eur_mw_h]),
                float(row[self.schema.afrr_cap_down_eur_mw_h]),
                float(act_up_signal),
                float(act_down_signal),
                day_ahead_kw,
                id_adjustment_kw,
                day_ahead_kw + id_adjustment_kw,
                float(self.id_steps_left),
                float(self.block_steps_left),
                tod_sin,
                tod_cos,
                float(self.t / max(1, self.episode_len - 1)),
                float(self.d1_cap_up_kw),
                float(self.d1_cap_down_kw),
            ],
            dtype=np.float32,
        )

        return {
            "physical": physical,
            "forecast": np.array(self._forecast_matrix(), dtype=np.float32),
            "day_ahead_context": np.array(self._day_ahead_context_matrix(), dtype=np.float32),
            "action_mask": np.array(
                [
                    1.0 if self._day_ahead_due() else 0.0,
                    1.0 if self._intraday_due() else 0.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
        }

    def step(self, action: Dict[str, Any]):
        action = action or {}
        row = self._row(self.t)
        day_ahead_due_before = self._day_ahead_due()
        intraday_due_before = self._intraday_due()

        day_ahead_applied = self._maybe_commit_day_ahead_from_action(action)
        intraday_applied = self._maybe_update_intraday(action.get("intraday"))

        real_time_action = action.get("real_time", action.get("low_level", np.zeros(1, dtype=np.float32)))
        real_time = np.array(real_time_action, dtype=np.float32).reshape(-1)
        if real_time.size != 1:
            raise ValueError("action['real_time'] must contain [p_batt_base_kw] in the unified environment.")
        real_time = np.clip(real_time, self.action_space["real_time"].low, self.action_space["real_time"].high)

        dam_position_kw = float(self.day_ahead_schedule_kw[self.t])
        id_adjustment_kw = float(self.id_adjustment_profile_kw[self.t])
        reward, info = self._settle_step(
            row=row,
            dam_position_kw=dam_position_kw,
            id_adjustment_kw=id_adjustment_kw,
            p_batt_base_kw=float(real_time[0]),
            cap_up_req_kw=self.d1_cap_up_kw,
            cap_down_req_kw=self.d1_cap_down_kw,
        )
        info.update(
            {
                "market_mode": self.market_mode,
                "day_ahead_due": bool(day_ahead_due_before),
                "day_ahead_applied": bool(day_ahead_applied),
                "day_ahead_source": self.day_ahead_source,
                "intraday_due": bool(intraday_due_before),
                "intraday_applied": bool(intraday_applied),
                "intraday_gate_closure_steps": int(self.intraday_gate_steps),
                "intraday_target_start": int(self.last_intraday_target_start),
                "intraday_target_stop": int(self.last_intraday_target_stop),
                "day_ahead_schedule_kw": dam_position_kw,
                "current_id_offer_kw": float(self.current_id_offer_kw),
                "id_adjustment_kw": id_adjustment_kw,
                "d1_cap_up_kw": float(self.d1_cap_up_kw),
                "d1_cap_down_kw": float(self.d1_cap_down_kw),
                "realtime_trading_bias_kw": 0.0,
            }
        )
        self.log.append(info)

        self.block_steps_left = max(0, self.block_steps_left - 1)
        self.id_steps_left = max(0, self.id_steps_left - 1)
        self.t += 1
        terminated = self.t >= self.episode_len
        if terminated:
            next_obs = {
                "physical": np.zeros(self.observation_space["physical"].shape, dtype=np.float32),
                "forecast": np.zeros(self.observation_space["forecast"].shape, dtype=np.float32),
                "day_ahead_context": np.zeros(self.observation_space["day_ahead_context"].shape, dtype=np.float32),
                "action_mask": np.zeros(self.observation_space["action_mask"].shape, dtype=np.float32),
            }
        else:
            next_obs = self._get_obs()
        return next_obs, float(reward), terminated, False, info


class UnifiedPVBESSGreeceEnv(DayAheadPVBESSGreeceEnv):
    """Canonical Gymnasium environment for the PV+BESS Greek-market simulator.

    This is the main environment new work should target. It is one Gym env with
    one ``reset()``, one ``step()``, one ``action_space``, and one
    ``observation_space``. The market hierarchy is represented inside those
    spaces by multi-rate action blocks and an action mask:

    - ``day_ahead``: full delivery-day DAM schedule, committed once.
    - ``intraday``: additive IDM correction and aFRR capacity offers, updated
      on block boundaries and applied only after the gate-closure lead time.
    - ``real_time``: battery dispatch setpoint, applied every simulator step.

    The two levels of data are also unified in the observation:
    ``day_ahead_context`` gives the full-day scheduling view, while
    ``physical`` and ``forecast`` give the rolling operational view.
    """

    market_mode = "unified"

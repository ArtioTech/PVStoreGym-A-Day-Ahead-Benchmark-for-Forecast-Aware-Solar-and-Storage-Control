from __future__ import annotations

from pathlib import Path
from typing import Optional

from pv_bess_gym.data.signals import SignalSchema
from pv_bess_gym.envs.pv_bess_greece_env import (
    DayAheadMarketTiming,
    DayAheadPVBESSGreeceEnv,
    HierarchicalConfig,
    HierarchicalPVBESSGreeceEnv,
    MarketConfig,
    UnifiedPVBESSGreeceEnv,
)


def make_hierarchical_env(
    *,
    signals_path: str | Path,
    plant_config_path: str | Path,
    schema: Optional[SignalSchema] = None,
    tz: Optional[str] = "Europe/Athens",
    dt_hours: float = 0.25,
    afrr_block_minutes: int = 60,
    decision_block_minutes: int = 60,
    violation_penalty: float = 100.0,
    shortfall_penalty: float = 20.0,
    soc_init: float = 0.10,
    include_forecasts: bool = True,
    forecast_horizon_steps: int = 8,
    episode_len: Optional[int] = None,
) -> HierarchicalPVBESSGreeceEnv:
    """Build a hierarchical Gym environment from external signal and plant files."""
    from pv_bess_gym.config_io import load_plant_config
    from pv_bess_gym.data.signals import load_signals

    resolved_schema = schema or SignalSchema()
    signals = load_signals(signals_path, schema=resolved_schema, tz=tz)
    plant = load_plant_config(plant_config_path)

    market = MarketConfig(
        dt_h=dt_hours,
        afrr_block_minutes=afrr_block_minutes,
        violation_penalty_eur_per_kw=violation_penalty,
        fallback_shortfall_penalty_eur_per_mwh=shortfall_penalty,
    )
    hierarchy = HierarchicalConfig(decision_block_minutes=decision_block_minutes)
    resolved_episode_len = episode_len or int(round(24.0 / market.dt_h))

    return HierarchicalPVBESSGreeceEnv(
        signals=signals,
        schema=resolved_schema,
        plant=plant,
        market=market,
        hierarchy=hierarchy,
        episode_len=resolved_episode_len,
        soc_init=soc_init,
        include_forecasts=include_forecasts,
        forecast_horizon_steps=forecast_horizon_steps,
    )


def make_unified_env(
    *,
    signals_path: str | Path,
    plant_config_path: str | Path,
    schema: Optional[SignalSchema] = None,
    tz: Optional[str] = "Europe/Athens",
    dt_hours: float = 0.25,
    afrr_block_minutes: int = 60,
    intraday_block_minutes: int = 60,
    intraday_gate_closure_minutes: int = 60,
    violation_penalty: float = 100.0,
    shortfall_penalty: float = 20.0,
    soc_init: float = 0.10,
    include_forecasts: bool = True,
    forecast_horizon_steps: int = 8,
    episode_len: Optional[int] = None,
) -> UnifiedPVBESSGreeceEnv:
    """Build the canonical unified Gym environment.

    This is the preferred simulator interface for new algorithms. It exposes
    the day-ahead scheduling layer, intraday/reserve layer, and real-time
    battery layer through one Gym action/observation contract.
    """
    from pv_bess_gym.config_io import load_plant_config
    from pv_bess_gym.data.signals import load_signals

    resolved_schema = schema or SignalSchema()
    signals = load_signals(signals_path, schema=resolved_schema, tz=tz)
    plant = load_plant_config(plant_config_path)

    market = MarketConfig(
        dt_h=dt_hours,
        afrr_block_minutes=afrr_block_minutes,
        violation_penalty_eur_per_kw=violation_penalty,
        fallback_shortfall_penalty_eur_per_mwh=shortfall_penalty,
    )
    timing = DayAheadMarketTiming(
        intraday_block_minutes=intraday_block_minutes,
        intraday_gate_closure_minutes=intraday_gate_closure_minutes,
    )
    resolved_episode_len = episode_len or int(round(24.0 / market.dt_h))

    return UnifiedPVBESSGreeceEnv(
        signals=signals,
        schema=resolved_schema,
        plant=plant,
        market=market,
        timing=timing,
        episode_len=resolved_episode_len,
        soc_init=soc_init,
        include_forecasts=include_forecasts,
        forecast_horizon_steps=forecast_horizon_steps,
    )


def make_day_ahead_env(**kwargs) -> DayAheadPVBESSGreeceEnv:
    """Backwards-compatible alias for :func:`make_unified_env`."""
    return make_unified_env(**kwargs)

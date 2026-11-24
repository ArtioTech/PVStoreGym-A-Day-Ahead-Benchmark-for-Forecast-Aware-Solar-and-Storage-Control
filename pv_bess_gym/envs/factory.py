from __future__ import annotations

from pathlib import Path

import pandas as pd
from typing import Optional, Sequence

from pv_bess_gym.data.signals import SignalSchema
from pv_bess_gym.envs.pv_bess_greece_env import (
    DayAheadMarketTiming,
    DayAheadPVBESSGreeceEnv,
    HierarchicalConfig,
    HierarchicalPVBESSGreeceEnv,
    InformationConfig,
    MarketConfig,
    SimOptions,
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
    legacy_v04: bool = False,
    settlement: str = "v2",
    metering_losses: bool = True,
    battery_aux_drains: bool = True,
    directional_taper: bool = True,
    energy_limited_power: bool = True,
    calendar_days: bool = True,
    c_thr_eur_mwh: Optional[float] = None,
    pv_forecast_error_scale: float = 1.0,
    price_noise_sigma: float = 0.0,
    noise_seed: int = 0,
    info_filter: bool = False,
    da_price_info: str = "realised",
    da_forecaster: str = "prev_day",
    da_weather: str = "short_lead",
    id_weather: str = "same_as_da",
    short_lead_visible_h: float = 3.0,
    da_plant_consistent: bool = False,
    market_clock: str = "as_published",
    radiation_stamp: str = "as_published",
    id_reference: str = "pv_only",
    id_position_limit: str = "off",
    degradation_eur_per_kwh: Optional[float] = None,
    episode_days: Optional[Sequence[int]] = None,
    options: Optional[SimOptions] = None,
) -> UnifiedPVBESSGreeceEnv:
    """Build the canonical unified Gym environment.

    ``legacy_v04=True`` reproduces the v0.4 simulator (see :class:`SimOptions`);
    it overrides the individual behaviour flags ``settlement``,
    ``metering_losses``, ``battery_aux_drains``, ``directional_taper``,
    ``energy_limited_power`` and ``calendar_days``. ``options`` passes a
    complete :class:`SimOptions` and overrides all of them. ``c_thr_eur_mwh`` is the degradation cost per MWh of
    battery throughput ``|p_bat| dt`` (0 when omitted).
    ``pv_forecast_error_scale``, ``price_noise_sigma``,
    ``intraday_gate_closure_minutes`` and ``degradation_eur_per_kwh`` are the
    difficulty knobs used by the benchmark tiers. ``da_weather``, ``id_weather``
    and ``short_lead_visible_h`` set the weather forecast per decision layer
    (see :class:`InformationConfig`); ``da_plant_consistent`` makes the D-1
    optimisation of the baselines plan with the plant's energy accounting.
    ``market_clock`` and ``radiation_stamp`` correct the time stamps of the
    market and irradiance data (rev1, see :mod:`pv_bess_gym.data.time_alignment`).
    ``id_reference`` sets what the intraday baselines expect to deliver
    (``"pv_only"``: PV forecast, v3; ``"d1_plan"``: the D-1 planned metered
    injection updated by the PV forecast change, rev1 A1). ``id_position_limit``
    (``"off"``/``"clip"``) bounds the commercial position ``x + Y`` by the
    grid limits (rev1 A2). ``episode_days`` restricts
    the days a seeded ``reset()`` samples from (for example the train split).
    """
    from dataclasses import replace

    from pv_bess_gym.config_io import load_plant_config
    from pv_bess_gym.data.signals import load_signals
    from pv_bess_gym.data.time_alignment import align_time_stamps
    from pv_bess_gym.envs.information import load_information_data

    resolved_schema = schema or SignalSchema()
    signals = load_signals(signals_path, schema=resolved_schema, tz=tz)
    plant = load_plant_config(plant_config_path)
    info_data = load_information_data(signals_path, pd.DatetimeIndex(signals[resolved_schema.ts_col]))
    signals, info_data = align_time_stamps(signals, info_data, resolved_schema,
                                           market_clock=market_clock, radiation_stamp=radiation_stamp)
    if c_thr_eur_mwh is not None and degradation_eur_per_kwh is not None:
        raise ValueError("Give c_thr_eur_mwh or degradation_eur_per_kwh, not both.")
    if c_thr_eur_mwh is not None:
        degradation_eur_per_kwh = float(c_thr_eur_mwh) / 1000.0
    if degradation_eur_per_kwh is not None:
        plant.battery = replace(plant.battery, deg_eur_per_kwh_throughput=float(degradation_eur_per_kwh))

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

    env = UnifiedPVBESSGreeceEnv(
        signals=signals,
        schema=resolved_schema,
        plant=plant,
        market=market,
        timing=timing,
        episode_len=resolved_episode_len,
        soc_init=soc_init,
        include_forecasts=include_forecasts,
        forecast_horizon_steps=forecast_horizon_steps,
        options=options if options is not None else SimOptions.legacy() if legacy_v04 else SimOptions(
            settlement=settlement,
            metering_losses=metering_losses,
            battery_aux_drains=battery_aux_drains,
            directional_taper=directional_taper,
            energy_limited_power=energy_limited_power,
            calendar_days=calendar_days,
            id_position_limit=id_position_limit,
        ),
        information=InformationConfig(
            pv_forecast_error_scale=pv_forecast_error_scale,
            price_noise_sigma=price_noise_sigma,
            noise_seed=noise_seed,
            info_filter=info_filter,
            da_price_info=da_price_info,
            da_forecaster=da_forecaster,
            da_weather=da_weather,
            id_weather=id_weather,
            short_lead_visible_h=short_lead_visible_h,
        ),
        episode_days=episode_days,
        info_data=info_data,
    )
    # Read by the D-1 optimisation of the baselines (realism fix R4): plan with
    # the auxiliary load and the metering loss of the plant.
    env.da_plant_consistent = bool(da_plant_consistent)
    # Read by the intraday layer of the baselines (rev1 A1).
    if id_reference not in {"pv_only", "d1_plan"}:
        raise ValueError("id_reference must be 'pv_only' or 'd1_plan'.")
    env.id_reference = str(id_reference)
    env.market_clock, env.radiation_stamp = str(market_clock), str(radiation_stamp)
    return env


def make_day_ahead_env(**kwargs) -> DayAheadPVBESSGreeceEnv:
    """Backwards-compatible alias for :func:`make_unified_env`."""
    return make_unified_env(**kwargs)

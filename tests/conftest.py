"""Shared fixtures: a small synthetic market so tests need no downloaded data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pv_bess_gym.data.signals import SignalSchema
from pv_bess_gym.envs import (
    DayAheadMarketTiming,
    InformationConfig,
    MarketConfig,
    PlantConfig,
    SimOptions,
    UnifiedPVBESSGreeceEnv,
)
from pv_bess_gym.models.battery import BatteryConfig
from pv_bess_gym.models.grid import GridConfig
from pv_bess_gym.models.pv import PVConfig

DAM, IDP, IMB, CAP_UP, CAP_DN = 100.0, 90.0, 60.0, 8.0, 4.0


def make_signals(days: int = 4, *, sunny: bool = False, act_up: float = 0.0, act_down: float = 0.0) -> pd.DataFrame:
    """Flat-price winter days. ``sunny=False`` gives zero PV, so cash flows are exact."""
    ts = pd.date_range("2025-01-06", periods=96 * days, freq="15min", tz="Europe/Athens")
    hour = ts.hour + ts.minute / 60.0
    ghi = 600.0 * np.clip(np.sin(np.pi * (hour - 7.0) / 10.0), 0.0, None) if sunny else np.zeros(len(ts))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "ghi_wm2": ghi,
            "temp_air_c": 12.0,
            "wind_mps": 2.0,
            "ghi_fcst_wm2": ghi * 0.9,
            "temp_air_fcst_c": 12.0,
            "wind_fcst_mps": 2.0,
            "price_dam_eur_mwh": DAM,
            "price_id_eur_mwh": IDP,
            "price_imb_eur_mwh": IMB,
            "afrr_cap_up_eur_mw_h": CAP_UP,
            "afrr_cap_down_eur_mw_h": CAP_DN,
            "afrr_act_up": act_up,
            "afrr_act_down": act_down,
        }
    )


def make_env(
    signals: pd.DataFrame | None = None,
    *,
    legacy: bool = False,
    gate_minutes: int = 60,
    soc_init: float = 0.5,
    battery: BatteryConfig | None = None,
    grid: GridConfig | None = None,
    information: InformationConfig | None = None,
    options: SimOptions | None = None,
    info_data=None,
) -> UnifiedPVBESSGreeceEnv:
    plant = PlantConfig(pv=PVConfig(), battery=battery or BatteryConfig(), grid=grid or GridConfig())
    return UnifiedPVBESSGreeceEnv(
        signals=make_signals() if signals is None else signals,
        schema=SignalSchema(),
        plant=plant,
        market=MarketConfig(),
        timing=DayAheadMarketTiming(intraday_gate_closure_minutes=gate_minutes),
        soc_init=soc_init,
        options=options or (SimOptions.legacy() if legacy else SimOptions()),
        information=information,
        info_data=info_data,
    )


def action(env, *, dam=0.0, afrr=(0.0, 0.0), intraday=0.0, p_batt=0.0) -> dict:
    return {
        "day_ahead": np.full(env.episode_len, dam, dtype=np.float32),
        "day_ahead_afrr": np.array(afrr, dtype=np.float32),
        "intraday": np.array([intraday], dtype=np.float32),
        "real_time": np.array([p_batt], dtype=np.float32),
    }


@pytest.fixture
def env():
    return make_env()

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import make_signals
from pv_bess_gym.data.builders.auxiliary.io_utils import step_hold_to_15min
from pv_bess_gym.data.provenance import audit_signals, has_errors
from pv_bess_gym.data.signals import SignalSchema, load_signals

EXAMPLE = Path(__file__).resolve().parents[1] / "example_data"


def test_step_hold_does_not_interpolate_hourly_prices():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="Europe/Athens")
    hourly = pd.Series([10.0, 50.0], index=idx[::4])
    held, gaps = step_hold_to_15min(hourly, idx)
    assert held.tolist() == [10.0] * 4 + [50.0] * 4
    assert gaps == 0


def test_step_hold_reports_gaps():
    idx = pd.date_range("2025-01-01", periods=12, freq="15min", tz="Europe/Athens")
    series = pd.Series([10.0, 30.0], index=[idx[0], idx[8]])
    held, gaps = step_hold_to_15min(series, idx)
    assert gaps == 4
    assert not held.isna().any()


def test_legacy_example_data_is_flagged_as_proxy():
    df = load_signals(EXAMPLE, SignalSchema(), tz="Europe/Athens")
    findings = audit_signals(df)
    flagged = {f.column for f in findings if f.severity == "error"}
    assert {"price_dam_eur_mwh", "price_id_eur_mwh", "afrr_cap_up_eur_mw_h", "afrr_cap_down_eur_mw_h"} <= flagged


def test_independent_prices_pass_the_proxy_checks():
    df = make_signals(days=10)
    rng = np.random.default_rng(0)
    hours = np.repeat(rng.normal(100.0, 20.0, len(df) // 4), 4)
    df["price_dam_eur_mwh"] = hours
    df["price_id_eur_mwh"] = hours + np.repeat(rng.normal(0.0, 5.0, len(df) // 4), 4)
    df["price_imb_eur_mwh"] = rng.normal(100.0, 40.0, len(df))
    assert not has_errors(audit_signals(df))

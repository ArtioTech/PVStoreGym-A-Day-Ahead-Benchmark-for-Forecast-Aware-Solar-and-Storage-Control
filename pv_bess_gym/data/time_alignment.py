"""Time-stamp corrections applied when the signals are loaded (rev1 protocol).

Two flags, both off by default so protocols v3 and v4 are reproduced unchanged.

``market_clock``
    ``"as_published"`` (v3/v4): market values sit at the time stamps the
    dataset builder gave them. ``"cet"``: the builder read the HEnEx results
    files (``DELIVERY_MTU`` / ``SORT``) and the IPTO files (IMBABE ``STARTDATE``,
    ISP1 header) as Europe/Athens wall clock, but they are CET/CEST wall clock:
    on 2025-03-30 the files skip 02:00 and on 2025-10-26 they repeat 02:00
    (CET/CEST changes at 02:00 local, Athens at 03:00 local). CET/CEST is always
    one hour behind Athens, so every market value belongs to the period one
    hour *after* its stamp. With ``"cet"`` the market columns and the
    per-auction sidecar prices are moved one hour later on the continuous
    15-minute grid. The first hour of the data (2025-01-01 00:00-01:00 local,
    = 23:00-24:00 CET of 2024-12-31) is not in the dataset and becomes NaN;
    the environment skips days with missing market values. Publication times
    (``PUB_TIME``) are not moved (they are consistent with Athens time).

``radiation_stamp``
    ``"as_published"`` (v3/v4): the Open-Meteo hourly irradiance is used as if
    it were the value at its stamp. ``"interval_centre"``: Open-Meteo documents
    hourly solar radiation as the mean of the preceding hour (temperature and
    wind are instantaneous), so the irradiance columns (actual, short-lead
    forecast, Previous Runs ``_previous_day1`` / ``_previous_day2``) are moved
    30 minutes earlier, to the centre of their averaging interval. The last
    30 minutes of the data (2025-12-31 23:30-24:00, night) keep their own
    value; the loader checks that it is zero.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

MARKET_CLOCKS = ("as_published", "cet")
RADIATION_STAMPS = ("as_published", "interval_centre")
MARKET_SHIFT = pd.Timedelta(hours=1)          # CET/CEST -> EET/EEST
RADIATION_SHIFT = pd.Timedelta(minutes=30)    # end-of-hour mean -> interval centre


def _market_columns(schema: Any) -> list[str]:
    names = [schema.price_dam_eur_mwh, schema.price_id_eur_mwh, schema.price_imb_eur_mwh,
             schema.afrr_cap_up_eur_mw_h, schema.afrr_cap_down_eur_mw_h, schema.afrr_en_up_eur_mwh,
             schema.afrr_en_down_eur_mwh, schema.afrr_act_up, schema.afrr_act_down,
             getattr(schema, "penalty_shortfall_eur_mwh", None), getattr(schema, "penalty_unavailable_eur_mw", None)]
    return [n for n in names if n]


def _grid_steps(ts: pd.Series, shift: pd.Timedelta) -> int:
    """Rows to move for ``shift`` on a gap-free regular grid (raises otherwise)."""
    index = pd.DatetimeIndex(ts)
    steps = index[1:] - index[:-1]          # TimedeltaIndex, independent of the stored resolution
    if len(steps) == 0 or (steps != steps[0]).any():
        raise ValueError("Time-stamp corrections need a gap-free regular time grid.")
    step = pd.Timedelta(steps[0])
    n = shift / step
    if abs(n - round(n)) > 1e-9:
        raise ValueError(f"Shift {shift} is not a multiple of the grid step {step}.")
    return int(round(n))


def _later(values: np.ndarray, n: int) -> np.ndarray:
    """Value at row i becomes the value of row i - n (moved n rows later); first n rows NaN."""
    out = np.full(len(values), np.nan)
    out[n:] = np.asarray(values, dtype=float)[:-n]
    return out


def _earlier(values: np.ndarray, n: int) -> np.ndarray:
    """Value at row i becomes the value of row i + n; the last n rows keep their own value."""
    values = np.asarray(values, dtype=float)
    out = values.copy()
    out[:-n] = values[n:]
    return out


def align_time_stamps(
    signals: pd.DataFrame,
    info_data: Optional[Any],
    schema: Any,
    *,
    market_clock: str = "as_published",
    radiation_stamp: str = "as_published",
) -> tuple[pd.DataFrame, Optional[Any]]:
    """Return signals (and sidecar data) with the requested time-stamp corrections."""
    if market_clock not in MARKET_CLOCKS:
        raise ValueError(f"market_clock must be one of {MARKET_CLOCKS}.")
    if radiation_stamp not in RADIATION_STAMPS:
        raise ValueError(f"radiation_stamp must be one of {RADIATION_STAMPS}.")
    if market_clock == "as_published" and radiation_stamp == "as_published":
        return signals, info_data
    df = signals.copy()
    ts = df[schema.ts_col]

    if market_clock == "cet":
        n = _grid_steps(ts, MARKET_SHIFT)
        for col in _market_columns(schema):
            if col in df.columns:
                df[col] = _later(df[col].to_numpy(dtype=float), n)
        if info_data is not None:
            info_data.ida1 = _later(info_data.ida1, n)
            info_data.ida2 = _later(info_data.ida2, n)
            info_data.ida3 = _later(info_data.ida3, n)

    if radiation_stamp == "interval_centre":
        n = _grid_steps(ts, RADIATION_SHIFT)
        columns = [c for c in (schema.ghi_wm2, schema.ghi_fcst_wm2) if c and c in df.columns]
        tails = [df[c].to_numpy(dtype=float)[-n:] for c in columns]
        if info_data is not None and info_data.weather_lead is not None:
            tails += [info_data.weather_lead[k][-n:] for k in ("ghi_d1", "ghi_d2")]
        if any(np.any(np.abs(t) > 1e-9) for t in tails):
            raise ValueError("radiation_stamp='interval_centre' expects zero irradiance in the last 30 min of the data.")
        for col in columns:
            df[col] = _earlier(df[col].to_numpy(dtype=float), n)
        if info_data is not None and info_data.weather_lead is not None:
            for key in ("ghi_d1", "ghi_d2"):
                info_data.weather_lead[key] = _earlier(info_data.weather_lead[key], n)
    return df, info_data

"""Publication times and the information filter of the environment.

A controller may only observe a value that was public at decision time:

    value of signal i for delivery step k' is in the observation at step k
        only if  tau_i(k') <= k

Steps are counted from 00:00 of the delivery day; the D-1 decision is taken at
step 0, so everything published on D-1 has a negative publication step.

Publication rules
-----------------
``PUBLICATION_RULES`` states, per observed signal, where its publication time
comes from. Day-specific times (HEnEx auctions, IPTO ISP1) are read from
``publication_times_<year>.csv``; the others follow a fixed rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

NEVER = 10**9          # not published within the delivery day
ALWAYS = -(10**9)      # known before the delivery day by construction

PUBLICATION_RULES: dict[str, str] = {
    "dam": "HEnEx DAM results PUB_TIME of the delivery day (D-1, about 14:20-14:40)",
    "dam_forecast": "naive forecast from DAM prices of earlier delivery days, all published before D-1",
    "ida1": "HEnEx IDA1 results PUB_TIME (D-1, about 16:22)",
    "ida2": "HEnEx IDA2 results PUB_TIME (D-1, about 23:22)",
    "ida3": "HEnEx IDA3 results PUB_TIME (day D, about 11:22; delivery 12:00-24:00 only)",
    "isp1": "IPTO ISP1 run timestamp (D-1, about 16:15-20:40): aFRR capacity prices",
    "imbalance": "IPTO IMBABE weekly file, published days after the delivery week: never observable in-day",
    "weather_forecast": "short-lead Open-Meteo historical forecast shown to the D-1 layer (version-4 behaviour, da_weather=short_lead): ASSUMED issued before D-1; the series stitches the first hours of successive model runs and records no issue time",
    "weather_d2": "Open-Meteo Previous Runs `_previous_day2`: predicted 48 h before the valid time, so issued no later than 00:00 of D-2 for every delivery step (D-1 layer, da_weather=day_ahead)",
    "weather_d1": "Open-Meteo Previous Runs `_previous_day1`: predicted 24 h before the valid time (intraday fallback, id_weather=latest)",
    "weather_short_lead": "Open-Meteo historical forecast (first hours of runs updated every 6 h): ASSUMED observable from short_lead_visible_h (3 h) before the valid time (id_weather=latest)",
    "measurement": "measured at the current step: PV, weather, activation signal of step k published at k",
}


@dataclass
class InformationData:
    """Auction prices aligned with the signal rows, and publication times per day."""

    ida1: np.ndarray
    ida2: np.ndarray
    ida3: np.ndarray
    published: dict[tuple[str, str], pd.Timestamp]   # (delivery date ISO, market) -> time
    # Fixed-lead weather forecasts aligned with the signal rows (Open-Meteo
    # Previous Runs): keys ghi_d1, temp_air_d1, wind_d1, ghi_d2, temp_air_d2,
    # wind_d2. None when weather_previous_runs_<year>.csv is absent.
    weather_lead: Optional[dict[str, np.ndarray]] = None

    def publication_step(self, market: str, day_start: pd.Timestamp, dt_h: float) -> int:
        """First decision step at which ``market`` results of that day are public."""
        when = self.published.get((day_start.date().isoformat(), market))
        if when is None:
            return NEVER
        minutes = (when - day_start).total_seconds() / 60.0
        return int(np.ceil(minutes / (dt_h * 60.0)))


def load_information_data(signals_path: str | Path, timestamps: pd.DatetimeIndex) -> Optional[InformationData]:
    """Load the sidecar files that sit next to the signal files, if they exist."""
    root = Path(signals_path)
    root = root if root.is_dir() else root.parent
    price_files = sorted(root.glob("auction_prices_*.csv"))
    pub_files = sorted(root.glob("publication_times_*.csv"))
    if not price_files or not pub_files:
        return None
    prices = pd.concat([pd.read_csv(p) for p in price_files], ignore_index=True)
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    prices = prices.drop_duplicates("timestamp").set_index("timestamp")
    aligned = prices.reindex(pd.DatetimeIndex(timestamps).tz_convert("UTC"))
    pub = pd.concat([pd.read_csv(p) for p in pub_files], ignore_index=True)
    tz = timestamps.tz
    published = {
        (str(row.delivery_date), str(row.market)): pd.Timestamp(row.published_at).tz_convert(tz)
        for row in pub.itertuples(index=False)
    }

    def column(name: str) -> np.ndarray:
        return aligned[name].to_numpy(dtype=float) if name in aligned else np.full(len(aligned), np.nan)

    return InformationData(
        ida1=column("price_ida1_eur_mwh"),
        ida2=column("price_ida2_eur_mwh"),
        ida3=column("price_ida3_eur_mwh"),
        published=published,
        weather_lead=load_weather_lead(root, timestamps),
    )


WEATHER_LEAD_COLUMNS = ("ghi_d1", "temp_air_d1", "wind_d1", "ghi_d2", "temp_air_d2", "wind_d2")


def load_weather_lead(root: Path, timestamps: pd.DatetimeIndex) -> Optional[dict[str, np.ndarray]]:
    """Fixed-lead weather forecasts (scripts/data/fetch_open_meteo_previous_runs.py), aligned by timestamp."""
    files = sorted(Path(root).glob("weather_previous_runs_*.csv"))
    if not files:
        return None
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.drop_duplicates("timestamp").set_index("timestamp")
    aligned = frame.reindex(pd.DatetimeIndex(timestamps).tz_convert("UTC"))
    missing = int(aligned[list(WEATHER_LEAD_COLUMNS)].isna().sum().sum())
    if missing:
        raise ValueError(f"weather_previous_runs files leave {missing} values missing on the signal grid.")
    return {c: aligned[c].to_numpy(dtype=float) for c in WEATHER_LEAD_COLUMNS}


def naive_dam_forecast(timestamps: pd.DatetimeIndex, dam: np.ndarray, method: str) -> np.ndarray:
    """Naive DAM price forecast: the price at the same local time 1 or 7 days earlier.

    ``method`` is ``"prev_day"`` or ``"prev_week"``. Both use only prices of
    delivery days whose results were published before the D-1 decision of the
    target day. Where no earlier value exists (the first day or week of the
    data, and local times skipped by a DST change) the nearest earlier
    available lag is used, and for the very first day the realised price.
    """
    if method not in {"prev_day", "prev_week"}:
        raise ValueError("da_forecaster must be 'prev_day' or 'prev_week'.")
    local = pd.DatetimeIndex(timestamps).tz_localize(None)
    series = pd.Series(np.asarray(dam, dtype=float), index=local)
    series = series[~series.index.duplicated(keep="first")]
    lags = [7, 1] if method == "prev_week" else [1, 7]
    out = pd.Series(np.nan, index=local)
    for lag in lags + [2, 3, 14]:
        candidate = series.reindex(local - pd.Timedelta(days=lag)).to_numpy()
        out = out.where(out.notna(), pd.Series(candidate, index=local))
    out = out.where(out.notna(), pd.Series(np.asarray(dam, dtype=float), index=local))
    return out.to_numpy(dtype=float)

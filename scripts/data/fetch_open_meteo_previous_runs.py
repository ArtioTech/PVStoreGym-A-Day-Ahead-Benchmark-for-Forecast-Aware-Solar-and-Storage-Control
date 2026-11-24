"""Fetch archived weather forecasts with a fixed lead time (Open-Meteo Previous Runs API).

`<var>_previous_dayN` is the value predicted N x 24 h before its valid time
(Open-Meteo documentation, Previous Runs API). Fetches GHI (shortwave
radiation), 2 m temperature and 10 m wind for leads 1 and 2 days, hourly, UTC,
at the plant coordinates of the dataset builder, and writes one CSV on the
15-min grid of the processed dataset (time interpolation, as the builder does
for the other Open-Meteo series).

Usage:
    python scripts/data/fetch_open_meteo_previous_runs.py --year 2025
Output:
    data/raw/open_meteo/previous_runs_<year>_<month>.json (raw responses)
    data/processed/weather_previous_runs_<year>.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pv_bess_gym.data.builders.greek_2025 import Greek2025BuildConfig, TZ  # noqa: E402

ENDPOINT = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARS = {"shortwave_radiation": "ghi", "temperature_2m": "temp_air", "wind_speed_10m": "wind"}
LEADS = (1, 2)


def fetch(year: int, lat: float, lon: float, raw_dir: Path) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "pv-bess-gym/2.0"})
    hourly = [f"{v}_previous_day{d}" for d in LEADS for v in VARS]
    chunks = []
    for month in range(1, 13):
        start = pd.Timestamp(year=year, month=month, day=1, tz=TZ)
        stop = start + pd.offsets.MonthBegin(1)
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": (start.tz_convert("UTC") - pd.Timedelta(days=1)).date().isoformat(),
            "end_date": (stop.tz_convert("UTC") + pd.Timedelta(days=1)).date().isoformat(),
            "hourly": ",".join(hourly), "timezone": "GMT", "wind_speed_unit": "ms",
        }
        path = raw_dir / f"previous_runs_{year}_{month:02d}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            response = session.get(ENDPOINT, params=params, timeout=120)
            response.raise_for_status()
            payload = response.json()
            path.write_text(json.dumps(payload))
            time.sleep(1.0)
        h = payload["hourly"]
        idx = pd.DatetimeIndex(pd.to_datetime(h["time"], utc=True)).tz_convert(TZ)
        cols = {}
        for d in LEADS:
            for src, short in VARS.items():
                cols[f"{short}_d{d}"] = pd.to_numeric(pd.Series(h[f"{src}_previous_day{d}"]), errors="coerce").to_numpy()
        chunks.append(pd.DataFrame(cols, index=idx))
        meta = {k: payload.get(k) for k in ("latitude", "longitude", "elevation", "generationtime_ms")}
        print(month, meta, flush=True)
    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def to_grid(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=1, day=1, tz=TZ)
    target = pd.date_range(start, start + pd.DateOffset(years=1), freq="15min", inclusive="left", tz=TZ)
    full = df.reindex(df.index.union(target)).sort_index().interpolate(method="time").reindex(target)
    for c in full.columns:
        if c.startswith("ghi") or c.startswith("wind"):
            full[c] = full[c].clip(lower=0.0)
    return full


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    args = ap.parse_args()
    cfg = Greek2025BuildConfig()
    df = fetch(args.year, cfg.lat, cfg.lon, ROOT / "data/raw/open_meteo")
    print("hourly rows", len(df), "missing per column:", df.isna().sum().to_dict())
    grid = to_grid(df, args.year)
    print("15-min rows", len(grid), "missing after interpolation:", int(grid.isna().sum().sum()))
    out = ROOT / f"data/processed/weather_previous_runs_{args.year}.csv"
    grid.rename_axis("timestamp").to_csv(out)
    print("->", out)


if __name__ == "__main__":
    main()

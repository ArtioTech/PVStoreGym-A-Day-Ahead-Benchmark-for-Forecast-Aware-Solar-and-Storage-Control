from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

TABULAR_SUFFIXES = {".csv", ".xlsx", ".xls", ".tsv"}


@dataclass(frozen=True)
class ParsedSeries:
    """Normalized time series plus provenance metadata."""

    frame: pd.DataFrame
    source_path: Path
    source_kind: str
    notes: list[str]


def list_tabular_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIXES
    )


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".xlsx":
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                f"Reading Excel file '{path}' requires 'openpyxl'. Install project dependencies first."
            ) from exc
    if suffix == ".xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except ImportError as exc:
            raise ImportError(
                f"Reading legacy Excel file '{path}' requires 'xlrd'. Install project dependencies first."
            ) from exc
    raise ValueError(f"Unsupported file type: {path.suffix}")


def normalize_columns(columns: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for column in columns:
        clean = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        normalized.append(clean)
    return normalized


def maybe_parse_timestamp_frame(df: pd.DataFrame, tz: str) -> Optional[pd.DataFrame]:
    df = df.copy()
    df.columns = normalize_columns(df.columns)
    direct_candidates = [
        "timestamp",
        "datetime",
        "delivery_datetime",
        "market_time",
        "trading_period",
    ]
    for candidate in direct_candidates:
        if candidate in df.columns:
            ts = pd.to_datetime(df[candidate], errors="coerce")
            if ts.notna().any():
                if getattr(ts.dt, "tz", None) is None:
                    ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="infer")
                else:
                    ts = ts.dt.tz_convert(tz)
                df["timestamp"] = ts
                return df

    date_candidates = ["date", "delivery_date", "trading_date"]
    hour_candidates = ["hour", "delivery_hour", "trading_hour", "period", "mtu"]
    for date_col in date_candidates:
        if date_col not in df.columns:
            continue
        date_values = pd.to_datetime(df[date_col], errors="coerce")
        if date_values.isna().all():
            continue
        for hour_col in hour_candidates:
            if hour_col not in df.columns:
                continue
            hour_values = pd.to_numeric(df[hour_col], errors="coerce")
            if hour_values.isna().all():
                continue
            rounded = hour_values.dropna()
            if rounded.empty:
                continue
            if rounded.min() >= 1 and rounded.max() <= 24:
                ts = date_values + pd.to_timedelta(hour_values - 1, unit="h")
            elif rounded.min() >= 0 and rounded.max() <= 23:
                ts = date_values + pd.to_timedelta(hour_values, unit="h")
            elif rounded.min() >= 1 and rounded.max() <= 96:
                ts = date_values + pd.to_timedelta((hour_values - 1) * 15, unit="min")
            else:
                continue
            ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="infer")
            df["timestamp"] = ts
            return df
    return None


def pick_numeric_column(df: pd.DataFrame, patterns: Iterable[str]) -> Optional[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    for column in df.columns:
        text = str(column)
        if any(regex.search(text) for regex in compiled):
            series = pd.to_numeric(df[column], errors="coerce")
            if series.notna().any():
                return column
    return None


def upsample_to_15min(series: pd.Series, target_index: pd.DatetimeIndex, label: str) -> pd.Series:
    base = pd.to_numeric(series, errors="coerce")
    base.index = pd.DatetimeIndex(series.index)
    base = base[~base.index.duplicated(keep="last")].sort_index()
    inferred = pd.infer_freq(base.index)
    if inferred and inferred.upper().startswith("15"):
        out = base.reindex(target_index)
    elif inferred and inferred.upper().startswith("H"):
        out = base.reindex(base.index.union(target_index)).sort_index().ffill().reindex(target_index)
    else:
        out = base.reindex(base.index.union(target_index)).sort_index().interpolate(method="time").reindex(target_index)
    if out.isna().all():
        raise ValueError(f"Could not align parsed series '{label}' to 15-minute index.")
    return out


def step_hold_to_15min(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    max_hold_steps: int = 3,
) -> tuple[pd.Series, int]:
    """Align a market-price series to the 15-minute index without interpolating.

    Market prices are constant over their market time unit (60, 30 or 15
    minutes), so each published value is held forward for at most
    ``max_hold_steps`` extra 15-minute steps (3 covers a 60-minute unit).
    Linear interpolation would invent prices that never cleared.

    Steps still empty after the hold are genuine gaps in the source; they are
    forward- then back-filled and their count is returned so the caller can
    record it in the dataset metadata.
    """
    base = pd.to_numeric(series, errors="coerce").dropna()
    base.index = pd.DatetimeIndex(base.index)
    base = base[~base.index.duplicated(keep="last")].sort_index()
    held = base.reindex(base.index.union(target_index)).sort_index()
    if max_hold_steps > 0:
        held = held.ffill(limit=max_hold_steps)
    held = held.reindex(target_index)
    n_gaps = int(held.isna().sum())
    if n_gaps == len(target_index):
        raise ValueError("Could not align parsed price series to the 15-minute index.")
    if n_gaps:
        held = held.ffill().bfill()
    return held, n_gaps

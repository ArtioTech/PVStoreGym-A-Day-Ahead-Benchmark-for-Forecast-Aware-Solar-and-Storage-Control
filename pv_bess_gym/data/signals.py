from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import pandas as pd


@dataclass
class SignalSchema:
    """Column names used by the simulator input dataset."""

    ts_col: str = "timestamp"

    # Actual weather
    ghi_wm2: str = "ghi_wm2"
    temp_air_c: str = "temp_air_c"
    wind_mps: Optional[str] = "wind_mps"

    # Forecast weather
    ghi_fcst_wm2: Optional[str] = "ghi_fcst_wm2"
    temp_air_fcst_c: Optional[str] = "temp_air_fcst_c"
    wind_fcst_mps: Optional[str] = "wind_fcst_mps"

    # Market prices
    price_dam_eur_mwh: str = "price_dam_eur_mwh"
    price_id_eur_mwh: Optional[str] = "price_id_eur_mwh"
    price_imb_eur_mwh: str = "price_imb_eur_mwh"

    # Reserve prices
    afrr_cap_up_eur_mw_h: str = "afrr_cap_up_eur_mw_h"
    afrr_cap_down_eur_mw_h: str = "afrr_cap_down_eur_mw_h"
    afrr_en_up_eur_mwh: Optional[str] = "afrr_en_up_eur_mwh"
    afrr_en_down_eur_mwh: Optional[str] = "afrr_en_down_eur_mwh"

    # Reserve activations: either fraction in [0,1] or direct kW
    afrr_act_up: Optional[str] = "afrr_act_up"
    afrr_act_down: Optional[str] = "afrr_act_down"

    # Optional penalties supplied by the dataset
    penalty_shortfall_eur_mwh: Optional[str] = "penalty_shortfall_eur_mwh"
    penalty_unavailable_eur_mw: Optional[str] = "penalty_unavailable_eur_mw"


def _read_tabular_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data file type: {suffix}. Use CSV or Excel.")


SUPPORTED_SIGNAL_SUFFIXES = {".csv", ".xlsx", ".xls"}
MONTHLY_SIGNAL_FILE_RE = re.compile(r"^greek_\d{4}_\d{2}$", re.IGNORECASE)


def _resolve_signal_files(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SIGNAL_SUFFIXES
        )
        monthly_files = [candidate for candidate in files if MONTHLY_SIGNAL_FILE_RE.match(candidate.stem)]
        selected = monthly_files if monthly_files else files
        if selected:
            return selected
        raise FileNotFoundError(f"No CSV/XLSX signal files were found under directory: {path}")
    raise FileNotFoundError(f"Signal path does not exist: {path}")


LEGACY_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("datetime", "time"),
    "price_dam_eur_mwh": ("price_dam", "dam_price", "dam"),
    "price_id_eur_mwh": ("price_id", "id_price", "intraday_price", "idm_price"),
    "price_imb_eur_mwh": ("price_imb", "imb_price", "imbalance_price"),
    "afrr_cap_up_eur_mw_h": ("cap_afrr_up", "afrr_up_price", "reserve_up_price"),
    "afrr_cap_down_eur_mw_h": ("cap_afrr_down", "afrr_down_price", "reserve_down_price"),
    "afrr_en_up_eur_mwh": ("en_afrr_up",),
    "afrr_en_down_eur_mwh": ("en_afrr_down",),
    "afrr_act_up": ("act_afrr_up",),
    "afrr_act_down": ("act_afrr_down",),
}


def _apply_legacy_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, aliases in LEGACY_COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def load_signals(path: str | Path, schema: SignalSchema, tz: Optional[str] = None) -> pd.DataFrame:
    """Load simulator signals from CSV or Excel.

    The input may be a single CSV/Excel file or a directory of monthly
    CSV/Excel files. When a directory contains ``greek_YYYY_MM`` files, those
    are preferred and concatenated in filename order. Timestamps are parsed as
    UTC and optionally converted to a target timezone.
    """
    frames: list[pd.DataFrame] = []
    signal_files = _resolve_signal_files(path)
    for signal_file in signal_files:
        frame = _apply_legacy_aliases(_read_tabular_file(signal_file))
        if schema.ts_col not in frame.columns:
            raise ValueError(f"Missing required column: {schema.ts_col} in {signal_file}")
        frame[schema.ts_col] = pd.to_datetime(frame[schema.ts_col], utc=True, errors="coerce")
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    if tz is not None:
        df[schema.ts_col] = df[schema.ts_col].dt.tz_convert(tz)
    df = df.sort_values(schema.ts_col).reset_index(drop=True)
    if df[schema.ts_col].isna().any():
        raise ValueError("Some timestamps could not be parsed.")
    duplicate_mask = df[schema.ts_col].duplicated()
    if duplicate_mask.any():
        duplicate_ts = df.loc[duplicate_mask, schema.ts_col].iloc[0]
        raise ValueError(f"Duplicate timestamps detected after loading signal files; first duplicate: {duplicate_ts}")

    required = [
        schema.ts_col,
        schema.ghi_wm2,
        schema.temp_air_c,
        schema.price_dam_eur_mwh,
        schema.price_imb_eur_mwh,
        schema.afrr_cap_up_eur_mw_h,
        schema.afrr_cap_down_eur_mw_h,
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_signals_csv(path: str | Path, schema: SignalSchema, tz: Optional[str] = None) -> pd.DataFrame:
    """Backwards-compatible CSV loader."""
    return load_signals(path=path, schema=schema, tz=tz)

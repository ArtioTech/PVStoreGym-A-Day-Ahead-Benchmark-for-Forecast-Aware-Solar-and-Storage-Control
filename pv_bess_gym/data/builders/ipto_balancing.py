"""Parsers for IPTO (ADMIE) balancing-market publications.

Two public file families are used to replace the proxy/synthetic aFRR columns
of the v0.4 dataset with published data:

``ISP1ISPResults`` (daily, 30-minute dispatch periods)
    The ``aFRR`` sheet carries the D-1 Integrated Scheduling Process results,
    including the ``Price Up`` / ``Price Down`` balancing-capacity prices.
    ISP1 is cleared on D-1, which matches the D-1 aFRR commitment of the
    environment.

``BalancingEnergyProduct`` / ``BalancingCapacityProduct`` (weekly, 15-minute)
    System totals of activated aFRR energy [MWh] and awarded aFRR capacity
    [MW]. Their ratio is the system-wide utilisation of awarded aFRR capacity,
    which the simulator can apply pro rata to the plant's own commitment.
    EXPERIMENTAL: for 2025 the downward ratio exceeds 1 in about 17% of steps
    (activated downward energy is often larger than the awarded downward band),
    so it is not a clean utilisation figure. It is opt-in for that reason.

Assumptions that a user of the dataset should know about are written into the
``notes`` of every returned :class:`ParsedSeries` so they end up in the
per-month metadata files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pv_bess_gym.data.builders.auxiliary.io_utils import ParsedSeries, step_hold_to_15min

LOGGER = logging.getLogger(__name__)

TZ = "Europe/Athens"
_VERSION_RE = re.compile(r"_(\d{2})(?:_\d+)?\.xlsx?$", re.I)


def file_version(path: Path) -> int:
    """Publication version encoded in an IPTO filename (``..._02.xlsx`` -> 2)."""
    match = _VERSION_RE.search(path.name)
    return int(match.group(1)) if match else 0


def _localize(naive: pd.Series) -> pd.Series:
    """Localize naive Athens wall-clock timestamps; ambiguous DST hour -> NaT."""
    ts = pd.to_datetime(naive, errors="coerce", dayfirst=True, format="mixed")
    return ts.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")


def _latest_version(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-file frames and keep the newest version per timestamp."""
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    combined = combined.sort_values(["timestamp", "version"])
    return combined.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp").sort_index()


# ---------------------------------------------------------------------------
# ISP1 aFRR balancing-capacity prices
# ---------------------------------------------------------------------------

def _parse_isp_afrr_sheet(path: Path) -> Optional[pd.DataFrame]:
    try:
        workbook = pd.ExcelFile(path)
    except Exception as exc:  # unreadable download
        LOGGER.warning("Skipping unreadable ISP workbook %s: %s", path.name, exc)
        return None
    sheet = next((name for name in workbook.sheet_names if name.lower().endswith("afrr")), None)
    if sheet is None:
        return None
    raw = workbook.parse(sheet, header=None)
    labels = raw.iloc[:, 0].astype(str).str.strip().str.lower()

    def _row(label: str) -> Optional[int]:
        hits = labels[labels == label]
        return int(hits.index[0]) if len(hits) else None

    header_idx, up_idx, down_idx = _row("afrr"), _row("price up"), _row("price down")
    if header_idx is None or up_idx is None or down_idx is None:
        return None
    frame = pd.DataFrame(
        {
            "timestamp": _localize(raw.iloc[header_idx, 1:]),
            "up": pd.to_numeric(raw.iloc[up_idx, 1:], errors="coerce").to_numpy(),
            "down": pd.to_numeric(raw.iloc[down_idx, 1:], errors="coerce").to_numpy(),
        }
    ).dropna(subset=["timestamp"])
    frame["version"] = file_version(path)
    return frame if not frame.empty else None


def parse_isp1_afrr_capacity_prices(
    root: Path,
    target_index: pd.DatetimeIndex,
    year: int,
) -> Optional[dict[str, ParsedSeries]]:
    """Return real aFRR capacity prices from ISP1 result workbooks under ``root``."""
    paths = sorted(root.glob("*_ISP1ISPResults_*.xlsx"))
    frames = [frame for frame in (_parse_isp_afrr_sheet(path) for path in paths) if frame is not None]
    if not frames:
        return None
    combined = _latest_version(frames)
    combined = combined[combined.index.year == year]
    if combined.empty:
        return None

    out: dict[str, ParsedSeries] = {}
    for key, column in [("up", "afrr_cap_up_eur_mw_h"), ("down", "afrr_cap_down_eur_mw_h")]:
        aligned, n_filled = step_hold_to_15min(combined[key], target_index)
        out[column] = ParsedSeries(
            frame=pd.DataFrame({"value": aligned}, index=target_index),
            source_path=paths[0],
            source_kind="real",
            notes=[
                f"IPTO ISP1 results, aFRR sheet, 'Price {key.capitalize()}' row; {len(frames)} daily workbook(s), "
                "latest published version per dispatch period.",
                "ISP dispatch periods (30 minutes, 15 minutes from October 2025) held constant over the steps they cover.",
                "ASSUMPTION: the published value is used as EUR/MW per hour of awarded capacity. "
                "Verify the unit against the Balancing Market Rulebook before quoting capacity revenues.",
                f"{n_filled} of {len(target_index)} steps had no published value and were forward/back-filled.",
            ],
        )
    return out


# ---------------------------------------------------------------------------
# System aFRR activation ratio
# ---------------------------------------------------------------------------

def _read_weekly_product(path: Path, wanted: dict[str, str]) -> Optional[pd.DataFrame]:
    try:
        raw = pd.read_excel(path, header=0)
    except Exception as exc:
        LOGGER.warning("Skipping unreadable IPTO workbook %s: %s", path.name, exc)
        return None
    raw.columns = [str(column).strip() for column in raw.columns]
    start_col = next((c for c in raw.columns if c.upper() == "STARTDATE"), None)
    if start_col is None:
        return None
    frame = pd.DataFrame({"timestamp": _localize(raw[start_col])})
    for target, pattern in wanted.items():
        column = next((c for c in raw.columns if re.search(pattern, c, re.I)), None)
        if column is None:
            return None
        frame[target] = pd.to_numeric(raw[column], errors="coerce")
    frame["version"] = file_version(path)
    # Prefer .xlsx over .xls when both exist for the same version.
    frame["version"] = frame["version"] + (0.5 if path.suffix.lower() == ".xlsx" else 0.0)
    return frame.dropna(subset=["timestamp"])


def parse_system_afrr_activation(
    root: Path,
    target_index: pd.DatetimeIndex,
    year: int,
    dt_h: float = 0.25,
) -> Optional[dict[str, ParsedSeries]]:
    """Return the system aFRR activation ratio in [0, 1] for each direction.

    ratio = activated aFRR energy [MWh] / (awarded aFRR capacity [MW] * dt)
    """
    energy_paths = sorted(p for p in root.glob("*_BalancingEnergyProduct_*") if p.suffix.lower() in {".xls", ".xlsx"})
    capacity_paths = sorted(p for p in root.glob("*_BalancingCapacityProduct_*") if p.suffix.lower() in {".xls", ".xlsx"})
    energy_frames = [
        f for f in (
            _read_weekly_product(p, {"up": r"total\s+afrr\s+up", "down": r"total\s+afrr\s+dn"}) for p in energy_paths
        ) if f is not None
    ]
    capacity_frames = [
        f for f in (
            _read_weekly_product(p, {"up": r"afrrq_up", "down": r"afrrq_dn"}) for p in capacity_paths
        ) if f is not None
    ]
    if not energy_frames or not capacity_frames:
        return None
    energy = _latest_version(energy_frames)
    capacity = _latest_version(capacity_frames)
    energy = energy[energy.index.year == year]
    capacity = capacity[capacity.index.year == year]
    if energy.empty or capacity.empty:
        return None

    out: dict[str, ParsedSeries] = {}
    for key, column in [("up", "afrr_act_up"), ("down", "afrr_act_down")]:
        e = energy[key].abs().reindex(target_index)
        c = capacity[key].abs().reindex(target_index)
        ratio = e / (c * dt_h)
        ratio = ratio.where(c > 0.0)
        n_capped = int((ratio > 1.0).sum())
        n_missing = int(ratio.isna().sum())
        ratio = ratio.clip(lower=0.0, upper=1.0).fillna(0.0)
        out[column] = ParsedSeries(
            frame=pd.DataFrame({"value": ratio.astype(float)}, index=target_index),
            source_path=energy_paths[0],
            source_kind="derived",
            notes=[
                "System-wide aFRR utilisation: IPTO BalancingEnergyProduct activated aFRR energy divided by "
                "IPTO BalancingCapacityProduct awarded aFRR capacity times the 15-minute step.",
                "ASSUMPTION: the plant is activated pro rata to the system (same utilisation as the awarded fleet).",
                f"{n_capped} step(s) exceeded 1.0 and were capped; {n_missing} step(s) had no data or zero "
                "awarded capacity and were set to 0.",
            ],
        )
    return out


def activation_summary(series: pd.Series) -> dict[str, float]:
    """Small descriptive summary used in dataset metadata."""
    values = np.asarray(series, dtype=float)
    return {
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "share_above_0.1": float((values > 0.1).mean()),
    }

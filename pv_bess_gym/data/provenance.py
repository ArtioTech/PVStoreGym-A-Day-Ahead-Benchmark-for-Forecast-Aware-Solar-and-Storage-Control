"""Checks that tell published market data apart from proxy/synthetic stand-ins.

The v0.4 example dataset shipped DAM, intraday and aFRR capacity prices that
were *derived from the imbalance price* by the builder's fallback formulas,
while being described as real. These checks reproduce those formulas and flag
any column that matches them, plus a few physical sanity checks. They are used
by ``scripts/data/verify_dataset.py`` and by the test-suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pv_bess_gym.data.signals import SignalSchema


@dataclass(frozen=True)
class Finding:
    column: str
    severity: str  # "error" | "warning" | "ok"
    message: str


def _max_abs_diff(a: pd.Series, b: pd.Series) -> float:
    return float((a - b).abs().max())


def audit_signals(df: pd.DataFrame, schema: SignalSchema | None = None, tol: float = 1e-6) -> list[Finding]:
    """Return provenance and sanity findings for a loaded signal frame."""
    s = schema or SignalSchema()
    out: list[Finding] = []
    dam = df[s.price_dam_eur_mwh].astype(float)
    imb = df[s.price_imb_eur_mwh].astype(float)

    # --- v0.4 proxy formulas (see greek_2025._build_market_proxies) ---------
    dam_proxy = imb.rolling(16, min_periods=1).median().shift(4).bfill().ewm(alpha=0.08, adjust=False).mean()
    if _max_abs_diff(dam, dam_proxy) < tol:
        out.append(Finding(s.price_dam_eur_mwh, "error", "DAM price equals the smoothed-imbalance proxy; not HEnEx data."))
    else:
        out.append(Finding(s.price_dam_eur_mwh, "ok", "DAM price does not match the imbalance-derived proxy."))

    if s.price_id_eur_mwh and s.price_id_eur_mwh in df.columns:
        idp = df[s.price_id_eur_mwh].astype(float)
        if _max_abs_diff(idp, 0.65 * dam + 0.35 * imb) < tol:
            out.append(Finding(s.price_id_eur_mwh, "error", "Intraday price equals 0.65*DAM + 0.35*imbalance; not HEnEx data."))
        else:
            out.append(Finding(s.price_id_eur_mwh, "ok", "Intraday price does not match the DAM/imbalance blend proxy."))

    spread = (imb - dam).abs()
    up_proxy = 4.0 + 0.10 * spread + 0.08 * (imb - dam).clip(lower=0.0).rolling(8, min_periods=1).mean()
    down_proxy = 3.5 + 0.08 * spread + 0.06 * (dam - imb).clip(lower=0.0).rolling(8, min_periods=1).mean()
    for column, proxy in [(s.afrr_cap_up_eur_mw_h, up_proxy), (s.afrr_cap_down_eur_mw_h, down_proxy)]:
        if _max_abs_diff(df[column].astype(float), proxy.clip(lower=0.0)) < tol:
            out.append(Finding(column, "error", "aFRR capacity price equals the imbalance-spread proxy; not IPTO data."))
        else:
            out.append(Finding(column, "ok", "aFRR capacity price does not match the imbalance-spread proxy."))

    # --- market time unit: prices must be step functions, not interpolated --
    ts = pd.DatetimeIndex(df[s.ts_col]).tz_convert("UTC")
    hours = ts.floor("h")
    distinct = dam.groupby(hours).nunique()
    hourly_era = distinct[(distinct.index < pd.Timestamp("2025-10-01", tz="UTC"))]
    if len(hourly_era) and float((hourly_era > 1).mean()) > 0.05:
        out.append(Finding(
            s.price_dam_eur_mwh, "error",
            f"{(hourly_era > 1).mean():.0%} of hours before 2025-10-01 have more than one DAM price; "
            "an hourly product appears to have been interpolated.",
        ))

    # --- physical sanity -----------------------------------------------------
    if s.ghi_fcst_wm2 and s.ghi_fcst_wm2 in df.columns:
        dark = df[s.ghi_wm2].astype(float).rolling(9, center=True, min_periods=1).max() <= 0.0
        fcst = df[s.ghi_fcst_wm2].astype(float)
        share = float((fcst[dark] > 5.0).mean()) if dark.any() else 0.0
        if share > 0.02:
            out.append(Finding(s.ghi_fcst_wm2, "error", f"Forecast irradiance exceeds 5 W/m2 in {share:.0%} of dark steps; forecast looks synthetic."))
        if float(fcst.max()) > 1361.0:
            out.append(Finding(s.ghi_fcst_wm2, "error", f"Forecast irradiance peaks at {fcst.max():.0f} W/m2, above the solar constant."))
    if s.wind_mps and s.wind_mps in df.columns and float(df[s.wind_mps].mean()) > 5.0 and float(df[s.wind_mps].max()) > 20.0:
        out.append(Finding(s.wind_mps, "warning", f"Mean wind {df[s.wind_mps].mean():.1f} with max {df[s.wind_mps].max():.1f}: values look like km/h in a m/s column."))

    numeric = df.select_dtypes(include=[np.number])
    nans = numeric.isna().sum()
    for column, count in nans[nans > 0].items():
        out.append(Finding(str(column), "error", f"{int(count)} missing value(s)."))
    return out


def has_errors(findings: list[Finding]) -> bool:
    return any(f.severity == "error" for f in findings)

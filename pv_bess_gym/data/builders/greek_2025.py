from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import asdict, dataclass, field
from html import unescape
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urljoin, urlparse

import pandas as pd
import requests

from pv_bess_gym.data.builders.auxiliary.io_utils import (
    ParsedSeries,
    list_tabular_files,
    maybe_parse_timestamp_frame,
    normalize_columns,
    pick_numeric_column,
    read_tabular,
    upsample_to_15min,
)
from pv_bess_gym.data.builders.auxiliary.synthetic_signals import (
    build_afrr_activation_signals,
)

LOGGER = logging.getLogger(__name__)

TZ = "Europe/Athens"
USER_AGENT = "pv-bess-gym/0.4 greek-2025-builder"
CANONICAL_COLUMNS = [
    "timestamp",
    "ghi_wm2",
    "temp_air_c",
    "wind_mps",
    "ghi_fcst_wm2",
    "temp_air_fcst_c",
    "wind_fcst_mps",
    "price_dam_eur_mwh",
    "price_id_eur_mwh",
    "price_imb_eur_mwh",
    "afrr_cap_up_eur_mw_h",
    "afrr_cap_down_eur_mw_h",
    "afrr_act_up",
    "afrr_act_down",
]
REAL_COLUMNS = [
    "timestamp",
    "ghi_wm2",
    "temp_air_c",
    "wind_mps",
    "ghi_fcst_wm2",
    "temp_air_fcst_c",
    "wind_fcst_mps",
    "price_dam_eur_mwh",
    "price_id_eur_mwh",
    "price_imb_eur_mwh",
]
PROXY_COLUMNS = [
    "afrr_cap_up_eur_mw_h",
    "afrr_cap_down_eur_mw_h",
]
SYNTHETIC_COLUMNS = [
    "afrr_act_up",
    "afrr_act_down",
]


@dataclass
class Greek2025BuildConfig:
    """Configuration for the Greek 2025 dataset builder."""

    year: int = 2025
    month: Optional[int] = None
    lat: float = 37.98
    lon: float = 23.72
    timezone: str = TZ
    raw_henex: Path = Path("data/raw/henex")
    raw_admie: Path = Path("data/raw/admie")
    out_dir: Path = Path("data/processed")
    plant_config: Optional[Path] = None
    allow_missing_market_data: bool = False
    weather_timeout_s: int = 60
    seed: int = 2025
    save_metadata: bool = True
    download_public_data: bool = True


@dataclass
class ColumnSource:
    mode: str
    source: str
    notes: list[str] = field(default_factory=list)


def build_greek_2025_dataset(config: Greek2025BuildConfig) -> list[Path]:
    config = _resolve_location_defaults(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if config.download_public_data:
        download_public_market_data(
            year=config.year,
            month=config.month,
            raw_henex=config.raw_henex,
            raw_admie=config.raw_admie,
            timeout_s=config.weather_timeout_s,
        )

    weather = fetch_open_meteo_weather(
        year=config.year,
        lat=config.lat,
        lon=config.lon,
        timezone=config.timezone,
        timeout_s=config.weather_timeout_s,
    )
    weather_fcst = fetch_open_meteo_historical_forecasts(
        year=config.year,
        lat=config.lat,
        lon=config.lon,
        timezone=config.timezone,
        timeout_s=config.weather_timeout_s,
    )
    afrr_acts = build_afrr_activation_signals(weather.index, seed=config.seed + 29)

    market_frame, market_sources = build_market_frame(
        year=config.year,
        index=weather.index,
        raw_henex=config.raw_henex,
        raw_admie=config.raw_admie,
        allow_missing=config.allow_missing_market_data,
    )

    full = pd.concat([weather, weather_fcst, market_frame, afrr_acts], axis=1)
    full["timestamp"] = full.index
    full = full[CANONICAL_COLUMNS]

    created: list[Path] = []
    months = [config.month] if config.month else list(range(1, 13))
    for month in months:
        monthly = _slice_month(full, year=config.year, month=month, tz=config.timezone)
        validate_monthly_frame(monthly, allow_missing_market_data=config.allow_missing_market_data)
        csv_path = config.out_dir / f"greek_{config.year}_{month:02d}.csv"
        monthly_to_save = monthly.copy()
        monthly_to_save["timestamp"] = monthly_to_save["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        monthly_to_save.to_csv(csv_path, index=False)
        LOGGER.info(
            "Validation passed for %04d-%02d | rows=%d | start=%s | end=%s",
            config.year,
            month,
            len(monthly),
            monthly["timestamp"].iloc[0],
            monthly["timestamp"].iloc[-1],
        )
        created.append(csv_path)
        if config.save_metadata:
            meta_path = config.out_dir / f"greek_{config.year}_{month:02d}_metadata.json"
            metadata = build_month_metadata(config=config, month=month, market_sources=market_sources)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            created.append(meta_path)
    return created


def download_public_market_data(year: int, month: Optional[int], raw_henex: Path, raw_admie: Path, timeout_s: int) -> None:
    raw_henex.mkdir(parents=True, exist_ok=True)
    raw_admie.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT})
    try:
        henex = download_henex_publications(session, year=year, month=month, raw_dir=raw_henex, timeout_s=timeout_s)
        LOGGER.info("HEnEx downloader prepared %d file(s).", len(henex))
    except Exception as exc:
        LOGGER.warning("HEnEx public download failed gracefully: %s", exc)
    try:
        admie = download_admie_public_data(session, year=year, month=month, raw_dir=raw_admie, timeout_s=timeout_s)
        LOGGER.info("ADMIE downloader prepared %d file(s).", len(admie))
    except Exception as exc:
        LOGGER.warning("ADMIE public download failed gracefully: %s", exc)


def download_henex_publications(session: requests.Session, year: int, month: Optional[int], raw_dir: Path, timeout_s: int) -> list[Path]:
    downloaded: list[Path] = []
    downloaded.extend(_download_henex_annual_results_archive(session, raw_dir, year=year, timeout_s=timeout_s))
    if not downloaded:
        downloaded.extend(_download_henex_summary_series(session, raw_dir, year, month, timeout_s))
    return downloaded


def download_admie_public_data(session: requests.Session, year: int, month: Optional[int], raw_dir: Path, timeout_s: int) -> list[Path]:
    downloaded: list[Path] = []
    months = [month] if month else list(range(1, 13))
    for month_i in months:
        for flavor in ["day", "period"]:
            url = f"https://www.admie.gr/services/monadtimesefedreion-{flavor}.php?year={year}&month={month_i:02d}"
            response = session.get(url, timeout=timeout_s)
            if response.status_code == 200 and response.text.strip():
                path = raw_dir / f"admie_reserve_prices_{flavor}_{year}_{month_i:02d}.json"
                path.write_text(response.text, encoding="utf-8")
                downloaded.append(path)
    for category in ["IMBABE", "BalancingCapacityProduct", "BalancingEnergyProduct"]:
        for item in _fetch_admie_file_api(session, year, category, timeout_s):
            path = raw_dir / item["filename"]
            if not path.exists():
                response = session.get(item["url"], timeout=timeout_s)
                response.raise_for_status()
                path.write_bytes(response.content)
            downloaded.append(path)
    return downloaded


def fetch_open_meteo_weather(year: int, lat: float, lon: float, timezone: str, timeout_s: int) -> pd.DataFrame:
    LOGGER.info("Fetching Open-Meteo archive weather actuals for %s at lat=%.4f lon=%.4f", year, lat, lon)
    frame = _fetch_open_meteo_hourly(
        endpoint="https://archive-api.open-meteo.com/v1/archive",
        year=year,
        lat=lat,
        lon=lon,
        timezone=timezone,
        timeout_s=timeout_s,
        column_map={
            "shortwave_radiation": "ghi_wm2",
            "temperature_2m": "temp_air_c",
            "wind_speed_10m": "wind_mps",
        },
    )
    frame["ghi_wm2"] = frame["ghi_wm2"].clip(lower=0.0)
    return frame


def fetch_open_meteo_historical_forecasts(year: int, lat: float, lon: float, timezone: str, timeout_s: int) -> pd.DataFrame:
    LOGGER.info("Fetching Open-Meteo archived forecast weather for %s at lat=%.4f lon=%.4f", year, lat, lon)
    frame = _fetch_open_meteo_hourly(
        endpoint="https://historical-forecast-api.open-meteo.com/v1/forecast",
        year=year,
        lat=lat,
        lon=lon,
        timezone=timezone,
        timeout_s=timeout_s,
        column_map={
            "shortwave_radiation": "ghi_fcst_wm2",
            "temperature_2m": "temp_air_fcst_c",
            "wind_speed_10m": "wind_fcst_mps",
        },
    )
    frame["ghi_fcst_wm2"] = frame["ghi_fcst_wm2"].clip(lower=0.0)
    return frame


def _fetch_open_meteo_hourly(
    endpoint: str,
    year: int,
    lat: float,
    lon: float,
    timezone: str,
    timeout_s: int,
    column_map: dict[str, str],
) -> pd.DataFrame:
    local_start = pd.Timestamp(year=year, month=1, day=1, tz=timezone)
    local_end = pd.Timestamp(year=year + 1, month=1, day=1, tz=timezone)
    hourly_chunks: list[pd.DataFrame] = []
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT})
    for month in range(1, 13):
        month_start = pd.Timestamp(year=year, month=month, day=1, tz=timezone)
        next_month = month_start + pd.offsets.MonthBegin(1)
        utc_start = (month_start.tz_convert("UTC") - pd.Timedelta(days=1)).date().isoformat()
        utc_end = (next_month.tz_convert("UTC") + pd.Timedelta(days=1)).date().isoformat()
        response = session.get(
            endpoint,
            params={
                "latitude": lat,
                "longitude": lon,
                "start_date": utc_start,
                "end_date": utc_end,
                "hourly": ",".join(column_map.keys()),
                "timezone": "GMT",
                "wind_speed_unit": "ms",
            },
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly")
        if not hourly:
            raise ValueError(f"Open-Meteo response for {year}-{month:02d} did not contain hourly data.")
        hourly_index = pd.DatetimeIndex(pd.to_datetime(hourly["time"], utc=True)).tz_convert(timezone)
        chunk = pd.DataFrame({target: pd.to_numeric(hourly.get(source), errors="coerce") for source, target in column_map.items()}, index=hourly_index)
        hourly_chunks.append(chunk)
    hourly_df = pd.concat(hourly_chunks).sort_index()
    hourly_df = hourly_df[~hourly_df.index.duplicated(keep="last")]
    target_index = pd.date_range(start=local_start, end=local_end, freq="15min", inclusive="left", tz=timezone)
    full = hourly_df.reindex(hourly_df.index.union(target_index)).sort_index().interpolate(method="time")
    full = full.reindex(target_index)
    if full.isna().any().any():
        missing = full.columns[full.isna().any()].tolist()
        raise ValueError(f"Open-Meteo fetch succeeded but left missing values in {missing}.")
    return full


def build_market_frame(
    year: int,
    index: pd.DatetimeIndex,
    raw_henex: Path,
    raw_admie: Path,
    allow_missing: bool,
) -> tuple[pd.DataFrame, dict[str, ColumnSource]]:
    columns = [
        "price_dam_eur_mwh",
        "price_id_eur_mwh",
        "price_imb_eur_mwh",
        "afrr_cap_up_eur_mw_h",
        "afrr_cap_down_eur_mw_h",
    ]
    frame = pd.DataFrame(index=index, columns=columns, dtype=float)
    sources: dict[str, ColumnSource] = {}

    frame["price_dam_eur_mwh"], sources["price_dam_eur_mwh"] = _parse_henex_series(
        raw_henex,
        index,
        year=year,
        kind="dam",
        patterns=[r"(^|_)dam(_|$)", r"day.?ahead", r"clearing.?price"],
        price_patterns=[r"price", r"clearing", r"mwh"],
    )
    frame["price_id_eur_mwh"], sources["price_id_eur_mwh"] = _parse_henex_series(
        raw_henex,
        index,
        year=year,
        kind="id",
        patterns=[r"(^|_)id(m)?(_|$)", r"intraday", r"continuous", r"auction"],
        price_patterns=[r"price", r"clearing", r"mwh"],
    )
    frame["price_imb_eur_mwh"], sources["price_imb_eur_mwh"] = _parse_admie_series(
        raw_admie,
        index,
        year=year,
        target="price_imb_eur_mwh",
        file_patterns=[r"imb", r"balance", r"settlement"],
        column_patterns=[r"imb", r"price", r"settlement", r"mwh"],
    )
    frame["afrr_cap_up_eur_mw_h"], sources["afrr_cap_up_eur_mw_h"] = _parse_admie_series(
        raw_admie,
        index,
        year=year,
        target="afrr_cap_up_eur_mw_h",
        file_patterns=[r"afrr", r"reserve", r"balancing", r"up"],
        column_patterns=[r"up", r"afrr", r"reserve", r"price", r"mw_h"],
    )
    frame["afrr_cap_down_eur_mw_h"], sources["afrr_cap_down_eur_mw_h"] = _parse_admie_series(
        raw_admie,
        index,
        year=year,
        target="afrr_cap_down_eur_mw_h",
        file_patterns=[r"afrr", r"reserve", r"balancing", r"down"],
        column_patterns=[r"down", r"afrr", r"reserve", r"price", r"mw_h"],
    )

    reserve_snapshot = _parse_admie_reserve_snapshots(raw_admie, index, year)
    if reserve_snapshot is not None:
        for column in ["afrr_cap_up_eur_mw_h", "afrr_cap_down_eur_mw_h"]:
            if frame[column].isna().all():
                frame[column] = reserve_snapshot[column]
                sources[column] = ColumnSource(
                    mode="real",
                    source="ADMIE public reserve dashboard snapshot",
                    notes=["Official ADMIE dashboard JSON snapshot normalized to a 15-minute Europe/Athens series."],
                )

    proxy_frame, proxy_sources = _build_market_proxies(frame, index)
    for column, source in proxy_sources.items():
        if frame[column].isna().all():
            frame[column] = proxy_frame[column]
            sources[column] = source

    missing_cols = [column for column in columns if frame[column].isna().any()]
    if missing_cols and not allow_missing:
        raise FileNotFoundError(
            "Missing market data for columns "
            f"{missing_cols}. Populate '{raw_henex}' and '{raw_admie}' with parseable public files "
            "or rerun with --allow-missing-market-data to emit NaN placeholders intentionally."
        )
    return frame, sources


def _build_market_proxies(
    frame: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, ColumnSource]]:
    proxy = pd.DataFrame(index=index)
    sources: dict[str, ColumnSource] = {}

    imb = pd.to_numeric(frame["price_imb_eur_mwh"], errors="coerce")
    if imb.notna().any():
        dam_proxy = imb.rolling(16, min_periods=1).median().shift(4).bfill()
        dam_proxy = dam_proxy.ewm(alpha=0.08, adjust=False).mean()
        proxy["price_dam_eur_mwh"] = dam_proxy
        sources["price_dam_eur_mwh"] = ColumnSource(
            mode="proxy",
            source="Proxy from ADMIE imbalance price series",
            notes=["Constructed as a smoothed rolling-median proxy from the real ADMIE imbalance price when direct HEnEx DAM files are unavailable."],
        )

        id_proxy = 0.65 * dam_proxy + 0.35 * imb
        proxy["price_id_eur_mwh"] = id_proxy
        sources["price_id_eur_mwh"] = ColumnSource(
            mode="proxy",
            source="Proxy from ADMIE imbalance price and DAM proxy",
            notes=["Constructed as an intermediate-price proxy between the DAM proxy and the real ADMIE imbalance price when direct intraday files are unavailable."],
        )

        spread = (imb - dam_proxy).abs()
        up_signal = (imb - dam_proxy).clip(lower=0.0).rolling(8, min_periods=1).mean()
        down_signal = (dam_proxy - imb).clip(lower=0.0).rolling(8, min_periods=1).mean()
        proxy["afrr_cap_up_eur_mw_h"] = (4.0 + 0.10 * spread + 0.08 * up_signal).clip(lower=0.0)
        proxy["afrr_cap_down_eur_mw_h"] = (3.5 + 0.08 * spread + 0.06 * down_signal).clip(lower=0.0)
        sources["afrr_cap_up_eur_mw_h"] = ColumnSource(
            mode="proxy",
            source="Proxy from ADMIE imbalance-vs-DAM spread",
            notes=["Constructed as an upward aFRR capacity-price proxy from imbalance spread magnitude because a direct granular public reserve-capacity price series was not available."],
        )
        sources["afrr_cap_down_eur_mw_h"] = ColumnSource(
            mode="proxy",
            source="Proxy from ADMIE imbalance-vs-DAM spread",
            notes=["Constructed as a downward aFRR capacity-price proxy from imbalance spread magnitude because a direct granular public reserve-capacity price series was not available."],
        )
    return proxy, sources


def validate_monthly_frame(df: pd.DataFrame, allow_missing_market_data: bool) -> None:
    if df.columns.tolist() != CANONICAL_COLUMNS:
        raise ValueError("Monthly frame columns do not match canonical schema/order.")
    ts = pd.DatetimeIndex(df["timestamp"])
    if not ts.is_monotonic_increasing:
        raise ValueError("Timestamps are not monotonic increasing.")
    if ts.has_duplicates:
        raise ValueError("Duplicate timestamps detected.")
    expected = pd.date_range(start=ts[0], end=ts[-1], freq="15min", tz=ts.tz)
    if len(expected) != len(ts) or not expected.equals(ts):
        raise ValueError("Monthly output is missing timestamps or has discontinuities.")
    if (df["ghi_wm2"] < 0).any():
        raise ValueError("ghi_wm2 contains negative values.")
    for column in ["afrr_act_up", "afrr_act_down"]:
        if ((df[column] < 0) | (df[column] > 1)).any():
            raise ValueError(f"{column} must stay in [0, 1].")
    required_no_nan = REAL_COLUMNS + PROXY_COLUMNS + SYNTHETIC_COLUMNS
    if allow_missing_market_data:
        required_no_nan = [column for column in required_no_nan if column not in {
            "price_dam_eur_mwh",
            "price_id_eur_mwh",
            "price_imb_eur_mwh",
            "afrr_cap_up_eur_mw_h",
            "afrr_cap_down_eur_mw_h",
        }]
    missing = [column for column in required_no_nan if df[column].isna().any()]
    if missing:
        raise ValueError(f"Validation failed: NaNs present in {missing}.")


def build_month_metadata(
    config: Greek2025BuildConfig,
    month: int,
    market_sources: dict[str, ColumnSource],
) -> dict[str, Any]:
    column_meta: dict[str, Any] = {}
    for column in REAL_COLUMNS:
        if column == "timestamp":
            column_meta[column] = {"classification": "real", "source": config.timezone}
        elif column in {"ghi_wm2", "temp_air_c", "wind_mps"}:
            column_meta[column] = {
                "classification": "real",
                "source": "Open-Meteo archive API",
                "notes": ["Historical weather actuals at hourly resolution, interpolated to 15-minute Europe/Athens index."],
            }
        elif column in {"ghi_fcst_wm2", "temp_air_fcst_c", "wind_fcst_mps"}:
            column_meta[column] = {
                "classification": "real",
                "source": "Open-Meteo historical forecast API",
                "notes": ["Archived weather forecast series at hourly resolution, interpolated to 15-minute Europe/Athens index."],
            }
        else:
            source = market_sources[column]
            column_meta[column] = {
                "classification": "real" if source.mode == "real" else "proxy_or_missing",
                "source": source.source,
                "notes": source.notes,
            }
    for column in PROXY_COLUMNS:
        source = market_sources[column]
        column_meta[column] = {
            "classification": "real" if source.mode == "real" else "proxy_or_missing",
            "source": source.source,
            "notes": source.notes,
        }
    for column in SYNTHETIC_COLUMNS:
        source = "Seeded regime-based aFRR activation model"
        column_meta[column] = {"classification": "synthetic", "source": source}

    real_columns = [column for column, meta in column_meta.items() if meta["classification"] == "real"]
    proxy_columns = [column for column, meta in column_meta.items() if meta["classification"] == "proxy_or_missing"]
    synthetic_columns = [column for column, meta in column_meta.items() if meta["classification"] == "synthetic"]

    return {
        "dataset": f"greek_{config.year}_{month:02d}",
        "year": config.year,
        "month": month,
        "timezone": config.timezone,
        "lat": config.lat,
        "lon": config.lon,
        "allow_missing_market_data": config.allow_missing_market_data,
        "real_columns": real_columns,
        "proxy_columns": proxy_columns,
        "synthetic_columns": synthetic_columns,
        "columns": column_meta,
    }


def _resolve_location_defaults(config: Greek2025BuildConfig) -> Greek2025BuildConfig:
    if config.plant_config and config.plant_config.exists():
        plant_path = config.plant_config
    else:
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / "example_data" / "station_athens.json"
        plant_path = candidate if candidate.exists() else None

    if not plant_path:
        return config

    payload = json.loads(plant_path.read_text(encoding="utf-8"))
    pv = payload.get("pv", {})
    return Greek2025BuildConfig(
        **{
            **asdict(config),
            "lat": float(pv.get("latitude", config.lat)),
            "lon": float(pv.get("longitude", config.lon)),
            "timezone": str(pv.get("tz", config.timezone)),
            "plant_config": plant_path,
        }
    )


def _slice_month(df: pd.DataFrame, year: int, month: int, tz: str) -> pd.DataFrame:
    start = pd.Timestamp(year=year, month=month, day=1, tz=tz)
    end = start + pd.offsets.MonthBegin(1)
    expected_index = pd.date_range(start=start, end=end, freq="15min", inclusive="left", tz=tz)
    out = df.loc[(df.index >= start) & (df.index < end)].copy()
    out = out.reindex(expected_index)
    out["timestamp"] = out.index
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def _parse_henex_series(
    root: Path,
    target_index: pd.DatetimeIndex,
    year: int,
    kind: str,
    patterns: list[str],
    price_patterns: list[str],
) -> tuple[pd.Series, ColumnSource]:
    workbook_parsed = _scan_henex_workbooks(root, target_index, year=year, kind=kind)
    if workbook_parsed is not None:
        return workbook_parsed.frame["value"], ColumnSource(
            mode="real",
            source=str(workbook_parsed.source_path),
            notes=workbook_parsed.notes,
        )
    parsed = _scan_market_files(root, target_index, year=year, file_patterns=patterns, column_patterns=price_patterns)
    if parsed is None:
        return pd.Series(index=target_index, dtype=float), ColumnSource(
            mode="missing",
            source=f"No parseable HEnEx {kind.upper()} files found under {root}",
            notes=["Downloader attempted official EnExGroup publications first; local raw ingestion remains the primary reliable path."],
        )
    return parsed.frame["value"], ColumnSource(mode="real", source=str(parsed.source_path), notes=parsed.notes)


def _parse_admie_series(
    root: Path,
    target_index: pd.DatetimeIndex,
    year: int,
    target: str,
    file_patterns: list[str],
    column_patterns: list[str],
) -> tuple[pd.Series, ColumnSource]:
    if target == "price_imb_eur_mwh":
        workbook_parsed = _scan_admie_imbalance_workbooks(root, target_index, year=year)
        if workbook_parsed is not None:
            return workbook_parsed.frame["value"], ColumnSource(
                mode="real",
                source=str(workbook_parsed.source_path),
                notes=workbook_parsed.notes,
            )
    parsed = _scan_market_files(root, target_index, year=year, file_patterns=file_patterns, column_patterns=column_patterns)
    if parsed is None:
        return pd.Series(index=target_index, dtype=float), ColumnSource(
            mode="missing",
            source=f"No parseable ADMIE series found for {target} under {root}",
            notes=["Downloader attempted official ADMIE API/report sources first; local raw ingestion remains the reliable fallback."],
        )
    return parsed.frame["value"], ColumnSource(
        mode="proxy" if parsed.source_kind == "proxy" else "real",
        source=str(parsed.source_path),
        notes=parsed.notes,
    )


def _scan_market_files(
    root: Path,
    target_index: pd.DatetimeIndex,
    year: int,
    file_patterns: list[str],
    column_patterns: list[str],
) -> Optional[ParsedSeries]:
    files = list_tabular_files(root)
    if not files:
        return None
    for path in files:
        file_key = path.stem.lower()
        if not any(re.search(pattern, file_key) for pattern in file_patterns):
            continue
        try:
            raw = read_tabular(path)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable market file %s: %s", path, exc)
            continue
        normalized = maybe_parse_timestamp_frame(raw, TZ)
        if normalized is None or "timestamp" not in normalized.columns:
            continue
        normalized = normalized.copy()
        normalized.columns = normalize_columns(normalized.columns)
        value_col = pick_numeric_column(normalized, column_patterns)
        if value_col is None:
            continue
        frame = normalized[["timestamp", value_col]].dropna(subset=["timestamp"]).rename(columns={value_col: "value"})
        frame = frame[frame["timestamp"].dt.year == year]
        if frame.empty:
            continue
        series = frame.set_index("timestamp")["value"]
        aligned = upsample_to_15min(series, target_index, label=path.name)
        notes = [f"Parsed from local raw file '{path.name}'."]
        inferred = pd.infer_freq(series.index)
        if inferred and inferred.upper().startswith("H"):
            notes.append("Hourly market series forward-filled/interpolated to 15-minute index for simulator use.")
        return ParsedSeries(
            frame=pd.DataFrame({"value": aligned}, index=target_index),
            source_path=path,
            source_kind="real",
            notes=notes,
        )
    return None


def _download_henex_summary_series(
    session: requests.Session,
    raw_dir: Path,
    year: int,
    month: Optional[int],
    timeout_s: int,
) -> list[Path]:
    templates = [
        ("dam", "https://www.enexgroup.gr/documents/20126/366820/{date}_EL-DAM_ResultsSummary_EN_v{version}.xlsx"),
        ("ida1", "https://www.enexgroup.gr/documents/20126/3257255/{date}_EL-IDA1_ResultsSummary_EN_v{version}.xlsx"),
        ("ida2", "https://www.enexgroup.gr/documents/20126/3257274/{date}_EL-IDA2_ResultsSummary_EN_v{version}.xlsx"),
        ("ida3", "https://www.enexgroup.gr/documents/20126/3257528/{date}_EL-IDA3_ResultsSummary_EN_v{version}.xlsx"),
    ]
    downloaded: list[Path] = []
    start = pd.Timestamp(year=year, month=month or 1, day=1)
    end = (start + pd.offsets.MonthBegin(1) - pd.Timedelta(days=1)) if month else pd.Timestamp(f"{year}-12-31")
    for day in pd.date_range(start, end, freq="D"):
        date_str = day.strftime("%Y%m%d")
        for label, template in templates:
            filename_prefix = f"{date_str}_EL-{label.upper() if label == 'dam' else label.upper()}".replace("DAM", "DAM").replace("IDA1", "IDA1").replace("IDA2", "IDA2").replace("IDA3", "IDA3")
            existing = list(raw_dir.glob(f"{date_str}_EL-{'DAM' if label == 'dam' else label.upper()}_ResultsSummary_EN_v*.xlsx"))
            if existing:
                downloaded.extend(existing)
                continue
            for version in ["01", "02", "03"]:
                url = template.format(date=date_str, version=version)
                response = session.get(url, timeout=timeout_s)
                if response.status_code == 200 and response.content[:2] == b"PK":
                    filename = Path(urlparse(url).path).name
                    path = raw_dir / filename
                    path.write_bytes(response.content)
                    downloaded.append(path)
                    break
    return downloaded


def _download_henex_annual_results_archive(
    session: requests.Session,
    raw_dir: Path,
    year: int,
    timeout_s: int,
) -> list[Path]:
    archive_name = f"{year}_EL-DAM-IDAs_Results.zip"
    archive_path = raw_dir / archive_name
    if not archive_path.exists():
        document_path = _resolve_enex_search_document_path(session, archive_name, timeout_s)
        if document_path is None:
            return []
        response = session.get(urljoin("https://www.enexgroup.gr", document_path), timeout=max(timeout_s, 180))
        response.raise_for_status()
        archive_path.write_bytes(response.content)
    extracted = _extract_henex_archive(archive_path, raw_dir / archive_path.stem)
    return [archive_path, *extracted]


def _resolve_enex_search_document_path(session: requests.Session, filename: str, timeout_s: int) -> Optional[str]:
    response = session.get("https://www.enexgroup.gr/search", params={"q": filename}, timeout=timeout_s)
    response.raise_for_status()
    normalized_target = _normalize_search_label(filename)
    detail_url: Optional[str] = None
    for match in re.finditer(r'<a[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<label>.*?)</a>', response.text, re.I | re.S):
        label = " ".join(re.sub(r"<[^>]+>", " ", match.group("label")).split())
        if normalized_target not in _normalize_search_label(label):
            continue
        href = unescape(match.group("href")).replace("&amp;", "&")
        if "view_content.jsp" not in href:
            continue
        detail_url = urljoin("https://www.enexgroup.gr", href)
        break
    if detail_url is None:
        return None
    detail_response = session.get(detail_url, timeout=timeout_s)
    detail_response.raise_for_status()
    file_match = re.search(rf'(/documents/\d+/\d+/{re.escape(filename)})', detail_response.text, re.I)
    if file_match:
        return unescape(file_match.group(1))
    return None


def _extract_henex_archive(archive_path: Path, extract_root: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.lower().endswith(".xlsx"):
                continue
            member_parts = Path(member.filename).parts
            relative_parts = member_parts[1:] if member_parts and member_parts[0] == archive_path.name else member_parts
            relative_path = Path(*relative_parts)
            destination = extract_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with archive.open(member) as src, destination.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(destination)
    return extracted


def _normalize_search_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _download_enex_series(
    session: requests.Session,
    raw_dir: Path,
    page_url: str,
    filename_regex: str,
    year: int,
    timeout_s: int,
) -> list[Path]:
    html = session.get(page_url, timeout=timeout_s).text
    filename_pattern = filename_regex.strip("^$")
    sample_match = re.search(filename_pattern, html, re.I)
    if not sample_match:
        return []
    anchor_idx = html.find(sample_match.group(0))
    instance_matches = re.findall(r"INSTANCE_[A-Za-z0-9]+", html[max(0, anchor_idx - 3000):anchor_idx])
    if not instance_matches:
        return []
    instance = instance_matches[-1].replace("INSTANCE_", "")
    collected: list[Path] = []
    seen: set[str] = set()
    empty_pages = 0
    for cur in range(1, 90):
        current = _build_enex_page_url(page_url, instance, cur)
        page_html = session.get(current, timeout=timeout_s).text
        matches = _extract_enex_download_links(page_html, filename_regex)
        if not matches:
            empty_pages += 1
            if empty_pages >= 3:
                break
            continue
        empty_pages = 0
        older_found = False
        year_found = False
        for filename, href in matches:
            if filename in seen:
                continue
            seen.add(filename)
            file_year = int(filename[:4]) if filename[:4].isdigit() else None
            if file_year is not None and file_year < year:
                older_found = True
            if file_year != year:
                continue
            year_found = True
            path = raw_dir / filename
            if not path.exists():
                response = session.get(urljoin(page_url, href), timeout=timeout_s)
                response.raise_for_status()
                path.write_bytes(response.content)
            collected.append(path)
        if older_found and not year_found:
            break
    return collected


def _build_enex_page_url(page_url: str, instance: str, cur: int) -> str:
    params = {
        "p_p_id": f"com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_{instance}",
        "p_p_lifecycle": 0,
        "p_p_state": "normal",
        "p_p_mode": "view",
        f"_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_{instance}_delta": 7,
        "p_r_p_resetCur": "false",
        f"_com_liferay_asset_publisher_web_portlet_AssetPublisherPortlet_INSTANCE_{instance}_cur": cur,
    }
    return f"{page_url}?{urlencode(params)}"


def _extract_enex_download_links(html: str, filename_regex: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    rx = re.compile(
        r'<a href="(?P<href>[^"]*get_file[^"]+)"[^>]*>\s*<i class="icon-download"></i>\s*&nbsp;\s*(?P<filename>[^<]+)</a>',
        re.I,
    )
    wanted = re.compile(filename_regex.strip("^$"), re.I)
    for match in rx.finditer(html):
        filename = unescape(match.group("filename")).strip()
        if wanted.match(filename):
            out.append((filename, unescape(match.group("href"))))
    return out


def _fetch_admie_file_api(session: requests.Session, year: int, category: str, timeout_s: int) -> list[dict[str, str]]:
    response = session.get(
        "https://www.admie.gr/getOperationMarketFilewRange",
        params={
            "dateStart": f"{year}-01-01",
            "dateEnd": f"{year}-12-31",
            "FileCategory": category,
        },
        timeout=timeout_s,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError:
        return []
    items: list[dict[str, str]] = []
    for value in _walk_payload(payload):
        if isinstance(value, str) and (".xlsx" in value.lower() or ".csv" in value.lower()):
            url = value if value.startswith("http") else urljoin("https://www.admie.gr", value)
            items.append({"url": url, "filename": Path(urlparse(url).path).name})
        elif isinstance(value, dict):
            url = _find_download_url(value)
            if url:
                full_url = url if url.startswith("http") else urljoin("https://www.admie.gr", url)
                items.append({"url": full_url, "filename": Path(urlparse(full_url).path).name})
    unique = {item["filename"]: item for item in items if item["filename"]}
    return list(unique.values())


def _walk_payload(value: Any) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, dict):
        out.append(value)
        for nested in value.values():
            out.extend(_walk_payload(nested))
    elif isinstance(value, list):
        for nested in value:
            out.extend(_walk_payload(nested))
    else:
        out.append(value)
    return out


def _find_download_url(value: dict[str, Any]) -> Optional[str]:
    for item in value.values():
        if isinstance(item, str) and (".xlsx" in item.lower() or ".csv" in item.lower() or "/file-type/" in item):
            return item
    return None


def _scan_henex_workbooks(root: Path, target_index: pd.DatetimeIndex, year: int, kind: str) -> Optional[ParsedSeries]:
    frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    wanted = re.compile(r"^\d{8}_EL-DAM_(Results|ResultsSummary)_EN_v\d+\.xlsx$", re.I) if kind == "dam" else re.compile(
        r"^\d{8}_EL-(IDA\d?|IDM)_(Results|ResultsSummary)_EN_v\d+\.xlsx$",
        re.I,
    )
    for path in sorted(root.rglob("*.xlsx")):
        if not wanted.match(path.name):
            continue
        parsed = _parse_workbook_with_header_scan(
            path,
            price_patterns=[r"\bmcp\b", r"market.?clearing.?price", r"\bprice\b", r"clearing.?price"],
        )
        if parsed is None:
            continue
        daily = _build_daily_market_frame_from_filename(path.name, parsed)
        if daily is None or daily.empty:
            continue
        daily["priority"] = _henex_kind_priority(path.name)
        daily["version"] = _henex_version(path.name)
        frames.append(daily)
        source_paths.append(path)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["timestamp"].dt.year == year]
    if combined.empty:
        return None
    if kind == "id":
        combined = combined.sort_values(["timestamp", "priority", "version"]).drop_duplicates(subset=["timestamp"], keep="last")
        notes = [
            f"Parsed {len(source_paths)} downloaded HEnEx workbook(s); the canonical ID series uses the latest available official auction price per timestamp (IDA3 > IDA2 > IDA1, latest version preferred).",
        ]
    else:
        combined = combined.sort_values(["timestamp", "version"]).drop_duplicates(subset=["timestamp"], keep="last")
        notes = [f"Parsed {len(source_paths)} downloaded HEnEx workbook(s); latest file version preferred for duplicate timestamps."]
    series = combined.set_index("timestamp")["value"].sort_index()
    aligned = upsample_to_15min(series, target_index, label=f"HEnEx {kind}")
    return ParsedSeries(
        frame=pd.DataFrame({"value": aligned}, index=target_index),
        source_path=source_paths[0],
        source_kind="real",
        notes=notes,
    )


def _scan_admie_imbalance_workbooks(root: Path, target_index: pd.DatetimeIndex, year: int) -> Optional[ParsedSeries]:
    frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    for path in sorted(root.glob("*.xlsx")):
        if "imbabe" not in path.name.lower():
            continue
        parsed = _parse_workbook_with_header_scan(path, price_patterns=[r"imbalance.?price", r"settlement.?price", r"\bprice\b"])
        if parsed is None:
            continue
        parsed = parsed.copy()
        parsed.columns = normalize_columns(parsed.columns)
        if "startdate" in parsed.columns:
            timestamp = pd.to_datetime(parsed["startdate"], errors="coerce")
            if getattr(timestamp.dt, "tz", None) is None:
                timestamp = timestamp.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous=False)
            parsed["timestamp"] = timestamp
        else:
            normalized = maybe_parse_timestamp_frame(parsed, TZ)
            if normalized is None or "timestamp" not in normalized.columns:
                continue
            parsed = normalized
            parsed.columns = normalize_columns(parsed.columns)
        value_col = next((c for c in parsed.columns if "imbalance_price" in c), None)
        if value_col is None:
            value_col = pick_numeric_column(parsed, [r"imbalance.?price", r"settlement.?price", r"imbalance", r"price"])
        if value_col is None:
            continue
        frame = parsed[["timestamp", value_col]].rename(columns={value_col: "value"}).dropna()
        frames.append(frame)
        source_paths.append(path)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["timestamp"].dt.year == year]
    if combined.empty:
        return None
    series = combined.groupby("timestamp", as_index=True)["value"].mean().sort_index()
    aligned = upsample_to_15min(series, target_index, label="ADMIE IMBABE")
    return ParsedSeries(
        frame=pd.DataFrame({"value": aligned}, index=target_index),
        source_path=source_paths[0],
        source_kind="real",
        notes=["Parsed downloaded ADMIE activated balancing/settlement workbook(s) as imbalance price proxy."],
    )


def _parse_admie_reserve_snapshots(root: Path, target_index: pd.DatetimeIndex, year: int) -> Optional[pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for path in sorted(root.glob(f"admie_reserve_prices_*_{year}_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        frame = pd.json_normalize(payload)
        if frame.empty:
            continue
        frame.columns = normalize_columns(frame.columns)
        time_col = next((c for c in frame.columns if "time" in c or "date" in c), None)
        up_col = next((c for c in frame.columns if "up" in c and ("price" in c or "value" in c)), None)
        down_col = next((c for c in frame.columns if "down" in c and ("price" in c or "value" in c)), None)
        if not time_col or not up_col or not down_col:
            continue
        ts = pd.to_datetime(frame[time_col], errors="coerce", utc=True)
        if ts.notna().sum() == 0:
            ts = pd.to_datetime(frame[time_col], errors="coerce")
            if getattr(ts.dt, "tz", None) is None:
                ts = ts.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous=False)
        else:
            ts = ts.dt.tz_convert(TZ)
        records.append(
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "afrr_cap_up_eur_mw_h": pd.to_numeric(frame[up_col], errors="coerce"),
                    "afrr_cap_down_eur_mw_h": pd.to_numeric(frame[down_col], errors="coerce"),
                }
            ).dropna(subset=["timestamp"])
        )
    if not records:
        return None
    combined = pd.concat(records, ignore_index=True)
    combined = combined[combined["timestamp"].dt.year == year]
    if combined.empty:
        return None
    combined = combined.groupby("timestamp", as_index=True).mean(numeric_only=True).sort_index()
    out = pd.DataFrame(index=target_index)
    for column in ["afrr_cap_up_eur_mw_h", "afrr_cap_down_eur_mw_h"]:
        out[column] = upsample_to_15min(combined[column], target_index, label=column)
    return out


def _parse_workbook_with_header_scan(path: Path, price_patterns: list[str]) -> Optional[pd.DataFrame]:
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
    except Exception as exc:
        LOGGER.warning("Skipping workbook %s: %s", path, exc)
        return None
    for sheet_name, raw in sheets.items():
        header_row = _infer_header_row(raw, price_patterns)
        if header_row is None:
            continue
        try:
            df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
        except Exception:
            continue
        df.columns = normalize_columns(df.columns)
        if pick_numeric_column(df, price_patterns) is None:
            continue
        return df
    return None


def _infer_header_row(df: pd.DataFrame, price_patterns: list[str]) -> Optional[int]:
    for idx in range(min(len(df), 20)):
        values = [str(value).strip().lower() for value in df.iloc[idx].tolist()]
        text = " ".join(values)
        if any(re.search(pattern, text) for pattern in price_patterns) and any(
            token in text for token in ["time", "mtu", "period", "hour", "startdate", "enddate"]
        ):
            return idx
    return None


def _build_daily_market_frame_from_filename(filename: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    date_match = re.search(r"(\d{8})", filename)
    if not date_match:
        return None
    delivery_day = pd.Timestamp(date_match.group(1))
    price_col = pick_numeric_column(df, [r"\bmcp\b", r"market.?clearing.?price", r"\bprice\b"])
    if price_col is None:
        return None
    frame = df.copy()
    frame.columns = normalize_columns(frame.columns)
    if "bidding_zone_descr" in frame.columns:
        mainland_mask = frame["bidding_zone_descr"].astype(str).str.contains("mainland greece", case=False, na=False)
        if mainland_mask.any():
            frame = frame.loc[mainland_mask].copy()
    if frame.empty:
        return None
    if {"sort", "delivery_duration"}.issubset(frame.columns):
        sort_values = pd.to_numeric(frame["sort"], errors="coerce")
        duration_values = pd.to_numeric(frame["delivery_duration"], errors="coerce")
        if sort_values.notna().any() and duration_values.notna().any():
            start = delivery_day.tz_localize(TZ)
            timestamps = start + pd.to_timedelta((sort_values - 1) * duration_values, unit="min")
        else:
            timestamps = None
    elif "delivery_mtu" in frame.columns:
        timestamps = pd.to_datetime(frame["delivery_mtu"], errors="coerce")
        if getattr(timestamps.dt, "tz", None) is None:
            timestamps = timestamps.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous=False)
        else:
            timestamps = timestamps.dt.tz_convert(TZ)
    else:
        time_col = next((c for c in frame.columns if c in {"market_time_unit", "market_time", "time", "hour", "mtu", "period"}), None)
        if time_col is None:
            return None
        timestamps = _timestamps_from_mtu_column(delivery_day, frame[time_col])
    if timestamps is None:
        return None
    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "value": pd.to_numeric(frame[price_col], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        return None
    return out.groupby("timestamp", as_index=False)["value"].mean()


def _henex_kind_priority(filename: str) -> int:
    upper = filename.upper()
    if "IDA3" in upper:
        return 3
    if "IDA2" in upper:
        return 2
    if "IDA1" in upper:
        return 1
    return 0


def _henex_version(filename: str) -> int:
    match = re.search(r"_v(\d+)\.xlsx$", filename, re.I)
    return int(match.group(1)) if match else 0


def _timestamps_from_mtu_column(day: pd.Timestamp, values: pd.Series) -> Optional[pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        vmax = float(numeric.max())
        if vmax <= 24:
            ts = day + pd.to_timedelta(numeric - 1, unit="h")
        elif vmax <= 96:
            ts = day + pd.to_timedelta((numeric - 1) * 15, unit="min")
        else:
            return None
        return ts.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="infer")
    hhmm = pd.to_datetime(values.astype(str), format="%H:%M", errors="coerce")
    if hhmm.notna().any():
        base = day.tz_localize(TZ)
        return base + pd.to_timedelta(hhmm.dt.hour * 60 + hhmm.dt.minute, unit="min")
    return None

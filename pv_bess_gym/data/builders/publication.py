"""Per-auction prices and publication times, read from the raw publisher files.

The benchmark's information filter needs to know *when* each exogenous value
became public. Two sidecar files are written next to the monthly signal files:

``auction_prices_<year>.csv``
    15-minute series ``price_ida1_eur_mwh``, ``price_ida2_eur_mwh``,
    ``price_ida3_eur_mwh`` (NaN where an auction does not cover the step).
    The monthly files keep the *final* intraday price (IDA3 > IDA2 > IDA1).

``publication_times_<year>.csv``
    One row per delivery day and market (``DAM``, ``IDA1``, ``IDA2``, ``IDA3``,
    ``ISP1``) with ``published_at``: the ``PUB_TIME`` of the HEnEx results file,
    or the run timestamp in the header of the IPTO ISP1 workbook.

Signals without a row here follow fixed rules, see
:data:`pv_bess_gym.envs.information.PUBLICATION_RULES`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from pv_bess_gym.data.builders.auxiliary.io_utils import step_hold_to_15min

LOGGER = logging.getLogger(__name__)
TZ = "Europe/Athens"
AUCTIONS = ["DAM", "IDA1", "IDA2", "IDA3"]


def zone_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry the Greek zonal clearing price.

    Results workbooks also list cross-border schedules (``ASSET_DESCR`` such as
    ``GR-IT``); those rows can carry the neighbouring price, which differs from
    the zonal MCP by a cent in some periods. Only LOAD and SUPPLY rows are used.
    """
    df = df[df["BIDDING_ZONE_DESCR"].astype(str).str.contains("Mainland Greece", case=False)]
    if "ASSET_DESCR" in df.columns:
        internal = df[df["ASSET_DESCR"].astype(str).str.upper().isin(["LOAD", "SUPPLY"])]
        if not internal.empty:
            return internal
    return df


def _henex_day(path: Path) -> tuple[pd.Series, pd.Timestamp]:
    """15-minute prices of one results file, each MTU expanded over its own duration."""
    df = zone_rows(pd.read_excel(path))
    day = pd.Timestamp(path.name[:8]).tz_localize(TZ)
    per_mtu = df.groupby(["SORT", "DELIVERY_DURATION"], as_index=False)["MCP"].median()
    pieces = {}
    for row in per_mtu.itertuples(index=False):
        duration = int(row.DELIVERY_DURATION)
        start = day + pd.Timedelta(minutes=(int(row.SORT) - 1) * duration)
        for j in range(duration // 15):
            pieces[start + pd.Timedelta(minutes=15 * j)] = float(row.MCP)
    prices = pd.Series(pieces).sort_index()
    published = pd.to_datetime(df["PUB_TIME"]).max().tz_localize(TZ)
    return prices, published


def _isp1_published(path: Path) -> pd.Timestamp | None:
    book = pd.ExcelFile(path)
    sheet = next((name for name in book.sheet_names if name.lower().endswith("afrr")), None)
    if sheet is None:
        return None
    header = " ".join(str(v) for v in book.parse(sheet, header=None, nrows=1).iloc[0].tolist())
    match = re.search(r"ISP1_\d{4}-\d{2}-\d{2}_(\d{14})", header)
    return pd.Timestamp(match.group(1)).tz_localize(TZ) if match else None


def build_publication_sidecars(raw_henex: Path, raw_admie: Path, out_dir: Path, year: int) -> list[Path]:
    """Write the two sidecar files for ``year`` and return their paths."""
    index = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="15min", tz=TZ, inclusive="left")
    prices = pd.DataFrame(index=index)
    rows: list[dict] = []
    for market in AUCTIONS:
        pieces = []
        latest: dict[str, Path] = {}
        for path in sorted(Path(raw_henex).rglob(f"{year}????_EL-{market}_Results_EN_v*.xlsx")):
            latest[path.name[:8]] = path  # sorted(): highest version last
        for tag, path in latest.items():
            try:
                series, published = _henex_day(path)
            except Exception as exc:
                LOGGER.warning("Skipping %s: %s", path.name, exc)
                continue
            pieces.append(series)
            rows.append({"delivery_date": pd.Timestamp(tag).date().isoformat(), "market": market,
                         "published_at": published.isoformat(), "source_file": path.name})
        if market == "DAM" or not pieces:
            continue
        combined = pd.concat(pieces).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        prices[f"price_{market.lower()}_eur_mwh"] = combined.reindex(index)
    for path in sorted(Path(raw_admie).glob(f"{year}????_ISP1ISPResults_*.xlsx")):
        published = _isp1_published(path)
        if published is not None:
            rows.append({"delivery_date": pd.Timestamp(path.name[:8]).date().isoformat(), "market": "ISP1",
                         "published_at": published.isoformat(), "source_file": path.name})

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_path = out_dir / f"auction_prices_{year}.csv"
    table = prices.copy()
    table.insert(0, "timestamp", table.index.strftime("%Y-%m-%dT%H:%M:%S%z"))
    table.to_csv(prices_path, index=False)
    pub = pd.DataFrame(rows).sort_values(["delivery_date", "market", "published_at"])
    pub = pub.drop_duplicates(subset=["delivery_date", "market"], keep="last")
    pub_path = out_dir / f"publication_times_{year}.csv"
    pub.to_csv(pub_path, index=False)
    return [prices_path, pub_path]

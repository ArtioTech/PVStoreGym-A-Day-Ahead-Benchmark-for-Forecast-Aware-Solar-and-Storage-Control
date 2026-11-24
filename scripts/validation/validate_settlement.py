"""Check the simulator's settlement against the publishers' raw files.

For each selected delivery day a **prescribed schedule** (fixed in this file,
independent of any controller) is run through the environment of task
T1-Nominal. Its physical and commercial quantities (DAM schedule, intraday trades, metered
energy, aFRR capacity and activated energy) are then re-settled *independently*:
prices are read straight from the raw HEnEx and IPTO workbooks under
``data/raw`` with a separate parser, and each leg is recomputed from the
Greek market settlement formulas:

    DAM            : Q_dam * MCP_dam
    Intraday       : Q_id  * MCP_ida   (latest auction covering the period)
    Imbalance      : (metered - Q_dam - Q_id - activated balancing energy) * P_imb
    aFRR capacity  : awarded MW * capacity price * hours
    aFRR energy    : +up energy * P_up, -down energy * P_down
                     (P_up = P_down = P_imb unless the dataset has energy prices)

The script reports, per day and per leg, the simulator value, the independent
value and their difference. A non-zero difference means the processed dataset
or the simulator disagrees with the published prices.

Scope: this validates the settlement arithmetic and the whole data pipeline
(raw file -> processed CSV -> environment) against published prices. It is not
a comparison with a settlement statement issued to a real plant, which also
contains uplift, non-compliance and fee items the simulator does not model.

    python scripts/validation/validate_settlement.py          # = make conformance
    python scripts/validation/validate_settlement.py --days 2025-01-15 2025-10-15
    python scripts/validation/validate_settlement.py --algorithm da-opt-mpc   # a controller's schedule instead

Output: results/<protocol subdir>/conformance.json (and .csv).
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym import build_registered_algorithm, evaluate_agent  # noqa: E402
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import make_task_env, results_root  # noqa: E402

TZ = "Europe/Athens"
# Wall clock of the HEnEx/IPTO files: Athens time ("as_published", protocols up to v3) or CET/CEST
# ("cet", protocol rev1). Set from env.market_clock in main().
RAW_CLOCK = {"as_published": TZ, "cet": "Europe/Brussels"}
CLOCK = TZ
DT = 0.25
DEFAULT_DAYS = ["2025-01-15", "2025-04-10", "2025-07-22", "2025-10-15", "2025-12-03"]


def _index(day: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz=TZ)
    return pd.date_range(start, start + pd.Timedelta(days=1), freq="15min", inclusive="left")


def _hold(series: pd.Series, index: pd.DatetimeIndex, steps: int) -> np.ndarray:
    series = series[~series.index.duplicated(keep="last")].sort_index()
    out = series.reindex(series.index.union(index)).sort_index()
    if steps:
        out = out.ffill(limit=steps)
    return out.reindex(index).to_numpy(dtype=float)


def _henex_day(path: Path, day: str) -> pd.Series:
    """MCP of the Mainland Greece bidding zone, indexed by delivery start (Athens time)."""
    df = pd.read_excel(path)
    df = df[df["BIDDING_ZONE_DESCR"].astype(str).str.contains("Mainland Greece", case=False)]
    internal = df[df["ASSET_DESCR"].astype(str).str.upper().isin(["LOAD", "SUPPLY"])]
    df = internal if not internal.empty else df  # cross-border rows can carry the neighbouring price
    start = pd.Timestamp(day, tz=CLOCK)
    ts = start + pd.to_timedelta((df["SORT"].astype(int) - 1) * df["DELIVERY_DURATION"].astype(int), unit="min")
    ts = pd.DatetimeIndex(ts).tz_convert(TZ)
    price = pd.Series(df["MCP"].astype(float).to_numpy(), index=ts).groupby(level=0).median()
    duration = pd.Series(df["DELIVERY_DURATION"].astype(int).to_numpy(), index=ts).groupby(level=0).max()
    # Expand each delivery period into its 15-min steps, so a value never reaches past its own period.
    steps = [(t + pd.Timedelta(minutes=15 * j), v) for t, v in price.items() for j in range(max(int(duration[t]) // 15, 1))]
    return pd.Series([v for _, v in steps], index=pd.DatetimeIndex([t for t, _ in steps])).groupby(level=0).median()


def _trading_days(day: str) -> list[str]:
    """HEnEx trading days whose delivery overlaps the Athens day (the previous one too under CET)."""
    days = [day]
    if CLOCK != TZ:
        days.insert(0, (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    return days


def _henex(raw: Path, day: str, product: str) -> pd.Series:
    parts = []
    for d in _trading_days(day):
        files = sorted((raw / "henex").rglob(f"{d.replace('-', '')}_EL-{product}_Results_EN_v*.xlsx"))
        if files:
            parts.append(_henex_day(files[-1], d))
    return pd.concat(parts) if parts else pd.Series(dtype=float)


def raw_dam(raw: Path, day: str) -> np.ndarray:
    series = _henex(raw, day, "DAM")
    if series.empty:
        raise FileNotFoundError(f"No HEnEx DAM results for {day}")
    return _hold(series, _index(day), 0)


def raw_intraday(raw: Path, day: str) -> np.ndarray:
    merged = pd.Series(dtype=float)
    for auction in ["IDA1", "IDA2", "IDA3"]:  # later auctions overwrite earlier ones
        series = _henex(raw, day, auction)
        if not series.empty:
            held = pd.Series(_hold(series, _index(day), 0), index=_index(day)).dropna()
            merged = held.combine_first(merged)
    if merged.empty:
        raise FileNotFoundError(f"No HEnEx intraday auction results for {day}")
    return merged.reindex(_index(day)).to_numpy(dtype=float)


def raw_imbalance(raw: Path, day: str) -> np.ndarray:
    best: dict[pd.Timestamp, Path] = {}
    for d in _trading_days(day):
        target = pd.Timestamp(d)
        for path in sorted((raw / "admie").glob("*_IMBABE_*.xlsx")):
            week = pd.Timestamp(path.name[:8])
            if week <= target < week + pd.Timedelta(days=7):
                best[week] = path  # sorted(): the highest version wins
    if not best:
        raise FileNotFoundError(f"No IPTO IMBABE workbook covers {day}")
    parts = []
    for path in best.values():
        df = pd.read_excel(path, sheet_name=0)
        column = next(c for c in df.columns if "imbalance price" in str(c).lower())
        ts = pd.to_datetime(df["STARTDATE"]).dt.tz_localize(CLOCK, nonexistent="NaT", ambiguous="NaT")
        part = pd.Series(df[column].astype(float).to_numpy(), index=pd.DatetimeIndex(ts))
        parts.append(part[part.index.notna()].tz_convert(TZ))
    return _hold(pd.concat(parts), _index(day), 0)


def raw_afrr_capacity(raw: Path, day: str) -> tuple[np.ndarray, np.ndarray]:
    parts: dict[str, list[pd.Series]] = {"price up": [], "price down": []}
    for d in _trading_days(day):
        files = sorted((raw / "admie").glob(f"{d.replace('-', '')}_ISP1ISPResults_*.xlsx"))
        if not files:
            continue
        book = pd.ExcelFile(files[-1])
        sheet = next(name for name in book.sheet_names if name.lower().endswith("afrr"))
        rawdf = book.parse(sheet, header=None)
        labels = rawdf.iloc[:, 0].astype(str).str.strip().str.lower()
        head = labels[labels == "afrr"].index[0]
        ts = pd.to_datetime(rawdf.iloc[head, 1:], errors="coerce").dt.tz_localize(CLOCK, nonexistent="NaT", ambiguous="NaT")
        for label in parts:
            row = labels[labels == label].index[0]
            series = pd.Series(pd.to_numeric(rawdf.iloc[row, 1:], errors="coerce").to_numpy(), index=pd.DatetimeIndex(ts))
            parts[label].append(series[series.index.notna()].dropna().tz_convert(TZ))
    if not parts["price up"]:
        raise FileNotFoundError(f"No IPTO ISP1 results for {day}")
    up, down = (_hold(pd.concat(parts[label]), _index(day), 3) for label in parts)
    return up, down


class PrescribedScheduleAgent:
    """A fixed plan, the same for every day, that exercises every settlement leg.

    - DAM: 150 kW from 08:00 to 18:00, 20 kW otherwise
    - aFRR capacity: 60 kW up, 40 kW down
    - intraday: +25 kW for every block from 12:00, -15 kW before
    - battery: charge 80 kW 10:00-14:00, discharge 80 kW 18:00-21:00, idle otherwise
    """

    name = "prescribed-schedule"

    def __init__(self, env):
        self.env = env

    def reset(self) -> None:
        pass

    def act(self, obs):
        t = int(self.env.t)
        hour = t / 4.0
        dam = np.where((np.arange(96) >= 32) & (np.arange(96) < 72), 150.0, 20.0).astype(np.float32)
        target_hour = (t + int(self.env.intraday_gate_steps)) / 4.0
        intraday = 25.0 if target_hour >= 12.0 else -15.0
        p_batt = -80.0 if 10.0 <= hour < 14.0 else (80.0 if 18.0 <= hour < 21.0 else 0.0)
        return {"day_ahead": dam, "day_ahead_afrr": np.array([60.0, 40.0], dtype=np.float32),
                "intraday": np.array([intraday], dtype=np.float32), "real_time": np.array([p_batt], dtype=np.float32)}


def validate_day(env, algorithm: str, day: str, raw: Path) -> dict:
    dates = [env.day_date(k) for k in range(env.num_days)]
    if day not in dates:
        raise ValueError(f"{day} is not a full delivery day in the dataset (DST change days are excluded).")
    agent = PrescribedScheduleAgent(env) if algorithm == "prescribed" else build_registered_algorithm(algorithm, env)
    log = evaluate_agent(env, agent, day_indices=[dates.index(day)], soc_init=0.5).combined_df

    dam, idp, imb = raw_dam(raw, day), raw_intraday(raw, day), raw_imbalance(raw, day)
    cup, cdn = raw_afrr_capacity(raw, day)
    k = DT / 1000.0
    q_dam, q_id = log["dam_position_kw"].to_numpy(), log["id_adjustment_kw"].to_numpy()
    act = log["act_up_kw"].to_numpy() - log["act_down_kw"].to_numpy()
    metered = log["metered_kw"].to_numpy()
    independent = {
        "dam_eur": k * q_dam * dam,
        "id_eur": k * q_id * idp,
        "imbalance_eur": k * (metered - q_dam - q_id - act) * imb,
        "afrr_capacity_eur": k * (log["cap_up_commit_kw"].to_numpy() * cup + log["cap_down_commit_kw"].to_numpy() * cdn),
        "afrr_activation_eur": k * act * imb,
    }
    legs = {}
    for leg, values in independent.items():
        sim = log[leg].to_numpy()
        diff = sim - values
        legs[leg] = {
            "simulator_eur": float(np.nansum(sim)),
            "independent_eur": float(np.nansum(values)),
            "deviation_eur": float(np.nansum(diff)),
            "max_step_deviation_eur": float(np.nanmax(np.abs(diff))),
            "steps_without_published_price": int(np.isnan(values).sum()),
        }
    total_sim = float(sum(v["simulator_eur"] for v in legs.values()))
    total_dev = float(sum(v["deviation_eur"] for v in legs.values()))
    return {
        "day": day,
        "algorithm": algorithm,
        "legs": legs,
        "total_simulator_eur": total_sim,
        "total_deviation_eur": total_dev,
        "relative_deviation": abs(total_dev) / max(1e-9, abs(total_sim)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", nargs="+", default=DEFAULT_DAYS, help="Delivery days, YYYY-MM-DD.")
    parser.add_argument("--algorithm", default="prescribed", help="'prescribed' (default) or a registered controller")
    parser.add_argument("--signals", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--plant-config", type=Path, default=ROOT / "example_data" / "station_athens.json")
    parser.add_argument("--out", type=Path, default=results_root() / "conformance.json")
    parser.add_argument("--tolerance-eur", type=float, default=0.01)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    import time

    started = time.time()
    env = make_task_env("T1-Nominal", signals_path=args.signals, plant_config_path=args.plant_config)
    global CLOCK
    CLOCK = RAW_CLOCK[getattr(env, "market_clock", "as_published")]
    print(f"raw market files read as {CLOCK} wall clock (market_clock = {getattr(env, 'market_clock', 'as_published')})")
    reports = [validate_day(env, args.algorithm, day, args.raw) for day in args.days]

    rows = []
    for report in reports:
        for leg, values in report["legs"].items():
            rows.append({"day": report["day"], "leg": leg, **values})
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    worst = max(abs(r["total_deviation_eur"]) for r in reports)
    payload = {
        "meta": run_metadata(started=started, command="make conformance", task="T1-Nominal", schedule=args.algorithm),
        "days": list(args.days),
        "max_abs_deviation_eur_per_leg_and_day": float(table["deviation_eur"].abs().max()),
        "max_abs_deviation_eur_per_step": float(table["max_step_deviation_eur"].max()),
        "max_abs_total_deviation_eur_per_day": float(worst),
        "steps_without_published_price": int(table["steps_without_published_price"].sum()),
        "tolerance_eur": args.tolerance_eur,
        "passed": bool(worst <= args.tolerance_eur),
        "reports": reports,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    table.to_csv(args.out.with_suffix(".csv"), index=False)

    print(table.round(4).to_string(index=False))
    print(f"\nLargest daily deviation: {worst:.4f} EUR (tolerance {args.tolerance_eur} EUR)")
    print(f"Saved {args.out} and {args.out.with_suffix('.csv')}")
    return 0 if worst <= args.tolerance_eur else 1


if __name__ == "__main__":
    raise SystemExit(main())

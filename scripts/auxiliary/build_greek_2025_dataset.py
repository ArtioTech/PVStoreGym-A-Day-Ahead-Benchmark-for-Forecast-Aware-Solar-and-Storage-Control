from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.data.builders import Greek2025BuildConfig, build_greek_2025_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a research-grade 2025 Greece PV+BESS benchmark dataset. "
            "Weather actuals and archived weather forecasts are real Open-Meteo data; "
            "aFRR activations remain synthetic; HEnEx/ADMIE market columns are "
            "ingested from local raw files when available."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="Dataset year. The intended benchmark year is 2025.")
    parser.add_argument("--month", type=int, choices=range(1, 13), help="Optional single month to build.")
    parser.add_argument("--lat", type=float, help="Latitude override. Defaults to the Athens example plant when available.")
    parser.add_argument("--lon", type=float, help="Longitude override. Defaults to the Athens example plant when available.")
    parser.add_argument(
        "--raw-henex",
        type=Path,
        default=ROOT / "data" / "raw" / "henex",
        help="Directory containing manually downloaded HEnEx CSV/XLSX files.",
    )
    parser.add_argument(
        "--raw-admie",
        type=Path,
        default=ROOT / "data" / "raw" / "admie",
        help="Directory containing ADMIE CSV/XLSX/tabular exports.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory where monthly CSV and metadata files are saved.",
    )
    parser.add_argument(
        "--plant-config",
        type=Path,
        default=ROOT / "example_data" / "station_athens.json",
        help="Optional plant JSON used to infer the default Athens site coordinates.",
    )
    parser.add_argument(
        "--allow-missing-market-data",
        action="store_true",
        help="Allow NaN placeholders for missing market columns instead of failing fast.",
    )
    parser.add_argument("--seed", type=int, default=2025, help="Seed for synthetic aFRR activations.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    config = Greek2025BuildConfig(
        year=args.year,
        month=args.month,
        lat=37.98 if args.lat is None else args.lat,
        lon=23.72 if args.lon is None else args.lon,
        raw_henex=args.raw_henex,
        raw_admie=args.raw_admie,
        out_dir=args.out,
        plant_config=args.plant_config,
        allow_missing_market_data=args.allow_missing_market_data,
        seed=args.seed,
    )
    created = build_greek_2025_dataset(config)
    print("\nCreated/modified files:")
    for path in created:
        print(path)
    print("\nHow to run:")
    print("python scripts/auxiliary/build_greek_2025_dataset.py --year 2025")
    print("python scripts/auxiliary/build_greek_2025_dataset.py --year 2025 --month 1")
    print("python scripts/auxiliary/build_greek_2025_dataset.py --year 2025 --month 7")
    print("python scripts/auxiliary/build_greek_2025_dataset.py --year 2025 --lat 37.98 --lon 23.72")
    print(
        "python scripts/auxiliary/build_greek_2025_dataset.py --year 2025 "
        "--raw-henex data/raw/henex --raw-admie data/raw/admie --out data/processed"
    )
    print(f"\nMonthly CSV files are saved under: {config.out_dir}")
    print("\nExample command to build all 12 months:")
    print("python scripts/auxiliary/build_greek_2025_dataset.py --year 2025")


if __name__ == "__main__":
    main()

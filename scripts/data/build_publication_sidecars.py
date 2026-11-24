"""Write auction prices and publication times next to the monthly signal files.

    python scripts/data/build_publication_sidecars.py --year 2025
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.data.builders.publication import build_publication_sidecars  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--raw-henex", type=Path, default=ROOT / "data" / "raw" / "henex")
    parser.add_argument("--raw-admie", type=Path, default=ROOT / "data" / "raw" / "admie")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    for path in build_publication_sidecars(args.raw_henex, args.raw_admie, args.out, args.year):
        print(path)


if __name__ == "__main__":
    main()

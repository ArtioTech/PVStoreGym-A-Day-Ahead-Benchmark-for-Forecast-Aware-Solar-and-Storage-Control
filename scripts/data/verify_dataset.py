"""Audit a signal dataset for proxy/synthetic columns and basic sanity.

    python scripts/data/verify_dataset.py data/processed
    python scripts/data/verify_dataset.py example_data      # v0.4 sample: fails

Exit code 1 if any check reports an error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.data.provenance import audit_signals, has_errors  # noqa: E402
from pv_bess_gym.data.signals import SignalSchema, load_signals  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("signals", type=Path, help="Signal CSV file or directory of monthly files.")
    args = parser.parse_args()

    schema = SignalSchema()
    df = load_signals(args.signals, schema=schema, tz="Europe/Athens")
    findings = audit_signals(df, schema)
    print(f"{args.signals}: {len(df)} rows, {df[schema.ts_col].iloc[0]} .. {df[schema.ts_col].iloc[-1]}")
    for finding in findings:
        print(f"  [{finding.severity.upper():7s}] {finding.column}: {finding.message}")
    return 1 if has_errors(findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

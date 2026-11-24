"""DA-ONLY on T1 test: version 4 vs version 3, and the value of D-1 information.

1. DA-ONLY daily profit by component under the version-4 results
   (results/benchmark_T1-Nominal.csv) and the version-3 run (results/v3/parts).
2. Value of information under the version-3 protocol: DA-ONLY re-run with
   (a) realised DAM prices, (b) short-lead weather, (c) both, against the
   version-3 default (naive DAM forecast, `previous_day2` weather).

Run after the version-3 baselines stage:
    PVBESS_PROTOCOL=configs/journal.yaml python scripts/diagnostics/t1_da_only_analysis.py --jobs 8
    -> results/v3/diagnostics/t1_da_only_analysis.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pv_bess_gym.agents.registry import build_registered_algorithm  # noqa: E402
from pv_bess_gym.benchmark.episodes import run_episode  # noqa: E402
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, results_root, split_days  # noqa: E402

COMPONENTS = ["dam_eur", "id_eur", "afrr_capacity_eur", "afrr_activation_eur", "imbalance_eur", "degradation_eur", "penalties_eur"]
VARIANTS = {
    "v3_default": {},
    "realised_dam_prices": {"da_price_info": "realised"},
    "short_lead_weather": {"da_weather": "short_lead"},
    "realised_prices_and_short_lead_weather": {"da_price_info": "realised", "da_weather": "short_lead"},
}


def run_chunk(overrides: dict, days: list[int]) -> list[dict]:
    warnings.filterwarnings("ignore")
    proto = load_protocol()
    env = make_task_env("T1-Nominal", **overrides)
    agent = build_registered_algorithm("da-only", env, {})
    return [run_episode(env, agent, d, s, float(proto["episode"]["soc_init"])) for d in days for s in proto["episode"]["activation_seeds"]]


def means(frame: pd.DataFrame) -> dict:
    out = {"profit_eur": float(frame.profit_eur.mean())}
    out.update({c: float(frame[c].mean()) for c in COMPONENTS})
    for c in ("scheduled_kwh", "metered_kwh", "imbalance_kwh"):
        if c in frame:
            out[c] = float(frame[c].mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    started = time.time()
    v4 = pd.read_csv(ROOT / "results" / "benchmark_T1-Nominal.csv")
    v4 = v4[(v4.controller == "da-only") & (v4.split == "test")]
    v3 = pd.concat([pd.read_csv(p) for p in sorted((results_root() / "parts" / "T1-Nominal" / "da-only").glob("test_*.csv"))])
    a, b = means(v4), means(v3)
    out = {"n_episodes": int(len(v3)),
           "version4_vs_version3": {"version4": a, "version3": b, "change": {k: b[k] - a[k] for k in a if k in b}}}
    days = split_days(make_task_env("T1-Nominal"), "test")
    chunks = [days[i:i + 8] for i in range(0, len(days), 8)]
    voi = {}
    for name, overrides in VARIANTS.items():
        rows = [r for part in Parallel(n_jobs=args.jobs)(delayed(run_chunk)(overrides, c) for c in chunks) for r in part]
        voi[name] = means(pd.DataFrame(rows))
    base = voi["v3_default"]
    out["value_of_information"] = {name: {"means": m, "minus_v3_default": {k: m[k] - base[k] for k in m}} for name, m in voi.items()}
    out["check_v3_default_equals_benchmark_run"] = bool(np.isclose(base["profit_eur"], b["profit_eur"]))
    out["meta"] = run_metadata(started=started, seed=load_protocol()["episode"]["activation_seeds"])
    path = results_root() / "diagnostics" / "t1_da_only_analysis.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()

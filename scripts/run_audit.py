"""C1 - conference configuration under settlement v1 and v2 -> results/audit.json.

    python scripts/run_audit.py            # = make audit

Runs DA-OPT-RBC, DA-OPT-RT-MPC and DA-OPT-MPC on the windows of
``configs/conf_v1.yaml`` (v0.4 simulator, v0.4 data, SOC carried within each
7-day window, one seed) twice: with the conference settlement (v1) and with
only the settlement switched to v2. Every other flag stays at its v0.4 value,
so the difference isolates the settlement correction.

The published conference totals (7836.3 / 7416.8 / 9093.6 EUR) cannot be
regenerated from the repository; the windows used here are stated in the
config and in the output.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402

COMPONENTS = ["dam_eur", "id_eur", "afrr_capacity_eur", "afrr_activation_eur", "imbalance_eur",
              "degradation_eur", "penalties_eur"]
PUBLISHED = {"da-opt-rbc": 7836.3, "da-opt-rt-mpc": 7416.8, "da-opt-mpc": 9093.6}


def run(conf: dict, settlement: str, controller: str, season: str, start_date: str) -> dict:
    warnings.filterwarnings("ignore")
    from pv_bess_gym import build_registered_algorithm, evaluate_agent, make_unified_env
    from pv_bess_gym.envs import SimOptions

    options = replace(SimOptions.legacy(), settlement=settlement)
    env = make_unified_env(signals_path=ROOT / conf["data"]["signals"], plant_config_path=ROOT / conf["data"]["plant"],
                           options=options, soc_init=float(conf["episode"]["soc_init"]))
    ts = env._ts
    first_row = int((ts >= pd.Timestamp(start_date, tz=ts.tz)).argmax())
    days = [first_row // 96 + k for k in range(int(conf["days_per_window"]))]   # v0.4 addresses raw 96-row slices
    result = evaluate_agent(env, build_registered_algorithm(controller, env), day_indices=days,
                            soc_init=float(conf["episode"]["soc_init"]), carry_soc_between_days=True)
    log = result.combined_df
    row = {"settlement": settlement, "controller": controller, "season": season, "start_date": start_date,
           "first_timestamp": log["timestamp"].iloc[0], "profit_eur": float(result.total_profit)}
    row.update({c: float(log[c].sum()) for c in COMPONENTS})
    return row


def main() -> None:
    started = time.time()
    conf = yaml.safe_load((ROOT / "configs" / "conf_v1.yaml").read_text(encoding="utf-8"))
    jobs = [(s, c, season, date) for s in ("v1", "v2") for c in conf["controllers"] for season, date in conf["windows"].items()]
    rows = pd.DataFrame(Parallel(n_jobs=6)(delayed(run)(conf, *job) for job in jobs))
    totals = rows.groupby(["settlement", "controller"])[["profit_eur", *COMPONENTS]].sum()
    table = []
    for controller in conf["controllers"]:
        v1, v2 = totals.loc[("v1", controller)], totals.loc[("v2", controller)]
        table.append({
            "controller": controller,
            "published_conference_total_eur": PUBLISHED[controller],
            "v1_total_eur": float(v1.profit_eur), "v2_total_eur": float(v2.profit_eur),
            "change_eur": float(v2.profit_eur - v1.profit_eur),
            "change_percent": float(100.0 * (v2.profit_eur - v1.profit_eur) / v1.profit_eur),
            "v1_components_eur": {c: float(v1[c]) for c in COMPONENTS},
            "v2_components_eur": {c: float(v2[c]) for c in COMPONENTS},
        })
    rank = lambda key: [r["controller"] for r in sorted(table, key=lambda r: -r[key])]  # noqa: E731
    payload = {
        "meta": run_metadata(started=started, command="make audit", seed=conf["seed"], config="configs/conf_v1.yaml"),
        "note": "Published conference totals could not be regenerated; windows below are those of configs/conf_v1.yaml.",
        "windows": conf["windows"], "days_per_window": conf["days_per_window"], "total_days": 4 * conf["days_per_window"],
        "ranking_v1": rank("v1_total_eur"), "ranking_v2": rank("v2_total_eur"),
        "ranking_changes": rank("v1_total_eur") != rank("v2_total_eur"),
        "totals": table, "per_window": rows.to_dict(orient="records"),
    }
    out = ROOT / "results" / "audit.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for r in table:
        print(f"{r['controller']:14s} v1 {r['v1_total_eur']:9.1f}  v2 {r['v2_total_eur']:9.1f}  change {r['change_eur']:+8.1f} "
              f"({r['change_percent']:+.1f}%)  published {r['published_conference_total_eur']}")
    print("ranking v1:", payload["ranking_v1"], "| v2:", payload["ranking_v2"], f"-> {out}")


if __name__ == "__main__":
    main()

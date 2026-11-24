"""Part 0.6 (rev1 brief), read-only: committed aFRR capacity of ORACLE and ORACLE-DELIVER, T1 test.

Re-solves the oracles with the protocol-v3 settings (time limit, MIP gap, epsilon)
and records r+ / r- of the incumbent plan, plus U / U_D to check that the
re-solve matches results/v3/oracle_T1-Nominal.csv.

    .venv/bin/python results/rev1/part0/oracle_reserve.py oracle --seeds 0 1 2 3 4 5 6 7 8 9
    .venv/bin/python results/rev1/part0/oracle_reserve.py deliver --seeds 0
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PVBESS_PROTOCOL", "configs/journal_v3.yaml")

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.tasks import Task, load_protocol, make_task_env, split_days  # noqa: E402

OUT = ROOT / "results" / "rev1" / "part0"


def job(kind: str, day: int, seed: int) -> dict:
    warnings.filterwarnings("ignore")
    from pv_bess_gym.oracle import solve_oracle
    task, proto = Task.load("T1-Nominal"), load_protocol()
    env = make_task_env(task, c_thr_eur_mwh=task.c_thr_eur_mwh)
    kw = dict(soc_init=float(proto["episode"]["soc_init"]), activation_seed=seed,
              time_limit_s=float(proto["oracle"]["time_limit_s"]), mip_rel_gap=float(proto["oracle"]["mip_rel_gap"]))
    if kind == "deliver":
        kw.update(allow_imbalance=False, deliver_epsilon_kwh=float(proto["oracle"]["deliver_epsilon_kwh"]))
    res = solve_oracle(env, day, **kw)
    return {"kind": kind, "day": day, "date": env.day_date(day), "activation_seed": seed,
            "bound": res.upper_bound_eur, "incumbent": res.objective_eur, "mip_gap": res.mip_gap,
            "r_up_kw": float(res.plan["cap_up_kw"][0]), "r_dn_kw": float(res.plan["cap_down_kw"][0]),
            "solve_seconds": res.solve_seconds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["oracle", "deliver"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--jobs", type=int, default=13)
    args = ap.parse_args()
    proto = load_protocol()
    env = make_task_env(Task.load("T1-Nominal"), c_thr_eur_mwh=0.0)
    days = split_days(env, "test", proto)
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(job)(args.kind, d, s) for d in days for s in args.seeds)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"oracle_reserve_{args.kind}.csv", index=False)
    ref = pd.read_csv(ROOT / "results" / "v3" / "oracle_T1-Nominal.csv")
    col = "U" if args.kind == "oracle" else "U_D"
    m = df.merge(ref[ref.split == "test"][["day", "activation_seed", col]], on=["day", "activation_seed"])
    print(f"{len(df)} solves; max |bound - results/v3 {col}| = {(m.bound - m[col]).abs().max():.4f} EUR; "
          f"mean r+ {df.r_up_kw.mean():.2f} kW, mean r- {df.r_dn_kw.mean():.2f} kW")


if __name__ == "__main__":
    main()

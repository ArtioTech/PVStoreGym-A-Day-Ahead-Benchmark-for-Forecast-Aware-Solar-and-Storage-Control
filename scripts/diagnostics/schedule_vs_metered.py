"""Scheduled vs metered energy per controller (Part 0 of the realism brief, v3).

For each episode (day x activation seed) of a split, sums per step:
- scheduled energy E_sch = (DAM position + intraday adjustment + a+ r+ - a- r-) dt
  (settlement v2 definition), and
- metered energy at the point of common coupling,
and counts steps with negative imbalance (metered below schedule, "short").

Usage:
    python scripts/diagnostics/schedule_vs_metered.py --task T1-Nominal --split test \
        --controllers da-only da-opt-mpc --jobs 13 --out results/diagnostics/schedule_vs_metered_T1.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pv_bess_gym.agents.registry import build_registered_algorithm  # noqa: E402
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import Task, load_protocol, make_task_env, split_days  # noqa: E402

TOL_KWH = 1e-6


def run_day(task_name: str, controller: str, day: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    task, proto = Task.load(task_name), load_protocol()
    env = make_task_env(task)
    agent = build_registered_algorithm(controller, env, {})
    dt = float(env.market.dt_h)
    soc = float(proto["episode"]["soc_init"])
    rows = []
    for seed in proto["episode"]["activation_seeds"]:
        agent.reset()
        obs, _ = env.reset(options={"day": int(day), "soc_init": soc, "activation_seed": int(seed)})
        sch = met = imb = 0.0
        n = short = longs = 0
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs))
            done = bool(term or trunc)
            e_sch = (info["scheduled_position_kw"] + info["act_up_kw"] - info["act_down_kw"]) * dt
            e_met = info["metered_kw"] * dt
            sch += e_sch
            met += e_met
            imb += info["imbalance_kwh"]
            n += 1
            short += int(info["imbalance_kwh"] < -TOL_KWH)
            longs += int(info["imbalance_kwh"] > TOL_KWH)
        rows.append({"day": day, "seed": int(seed), "scheduled_kwh": sch, "metered_kwh": met,
                     "imbalance_kwh": imb, "steps": n, "short_steps": short, "long_steps": longs})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1-Nominal")
    ap.add_argument("--split", default="test")
    ap.add_argument("--controllers", nargs="+", default=["da-only", "da-opt-mpc"])
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="results/diagnostics/schedule_vs_metered_T1.json")
    args = ap.parse_args()
    started = time.time()
    task, proto = Task.load(args.task), load_protocol()
    days = split_days(make_task_env(task), args.split, proto)
    out = {"task": args.task, "split": args.split, "n_days": len(days), "controllers": {}}
    for ctrl in args.controllers:
        parts = Parallel(n_jobs=args.jobs)(delayed(run_day)(args.task, ctrl, d) for d in days)
        rows = [r for p in parts for r in p]
        sch = np.array([r["scheduled_kwh"] for r in rows])
        met = np.array([r["metered_kwh"] for r in rows])
        steps = sum(r["steps"] for r in rows)
        out["controllers"][ctrl] = {
            "n_episodes": len(rows),
            "mean_daily_scheduled_kwh": float(sch.mean()),
            "mean_daily_metered_kwh": float(met.mean()),
            "mean_daily_metered_minus_scheduled_kwh": float((met - sch).mean()),
            "mean_daily_imbalance_kwh": float(np.mean([r["imbalance_kwh"] for r in rows])),
            "share_steps_negative_imbalance": sum(r["short_steps"] for r in rows) / steps,
            "share_steps_positive_imbalance": sum(r["long_steps"] for r in rows) / steps,
        }
        print(ctrl, json.dumps(out["controllers"][ctrl]), flush=True)
    out["meta"] = run_metadata(task=args.task, seed=list(proto["episode"]["activation_seeds"]), started=started)
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("->", path)


if __name__ == "__main__":
    main()

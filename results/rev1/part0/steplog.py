"""Part 0 (rev1 brief), read-only: per-step logs of every controller on T1-Nominal, test split.

Runs the controllers exactly as scripts/run_journal.py / scripts/rl/eval_rl.py do
(protocol v3, all days, all ten activation seeds), and records per step the
fields Part 0 needs. Changes no library code. Also returns the episode profit
so the run can be checked against results/v3/benchmark_T1-Nominal.csv.

    .venv/bin/python results/rev1/part0/steplog.py baselines
    .venv/bin/python results/rev1/part0/steplog.py rl --controllers da-opt-ppo da-opt-sac
    .venv-omnisafe/bin/python results/rev1/part0/steplog.py rl --controllers da-opt-ppol
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PVBESS_PROTOCOL", "configs/journal_v3.yaml")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pv_bess_gym import build_registered_algorithm  # noqa: E402
from pv_bess_gym.benchmark.tasks import Task, load_protocol, make_task_env, results_root, split_days  # noqa: E402

OUT = ROOT / "results" / "rev1" / "part0" / os.environ.get("STEPLOG_DIR", "steplogs")   # steplogs_rev1 for protocol rev1
TASK = "T1-Nominal"


def _run(env, agent, day, seed, soc, plan_box):
    agent.reset()
    obs, _ = env.reset(options={"day": int(day), "soc_init": float(soc), "activation_seed": int(seed)})
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    i0 = u._i0
    ida = {"ida1": u.info_data.ida1[i0:i0 + u.episode_len], "ida2": u.info_data.ida2[i0:i0 + u.episode_len],
           "ida3": u.info_data.ida3[i0:i0 + u.episode_len]}
    settle_auction = np.where(~np.isnan(ida["ida3"]), "ida3", np.where(~np.isnan(ida["ida2"]), "ida2",
                              np.where(~np.isnan(ida["ida1"]), "ida1", "none")))
    seen_label = np.full(u.episode_len, "", dtype=object)
    seen_value = np.full(u.episode_len, np.nan)
    decided_at = np.full(u.episode_len, -1)
    rows, profit, done = [], 0.0, False
    while not done:
        if obs["action_mask"][1] > 0.5:
            prov = u.observation_provenance()
            lo = min(u.episode_len, u.t + u.intraday_gate_steps)
            hi = min(u.episode_len, lo + u.intraday_block_steps)
            seen_label[lo:hi] = prov["id_label"][lo:hi]
            seen_value[lo:hi] = prov["values"]["id"][lo:hi]
            decided_at[lo:hi] = u.t
        action = agent.act(obs)
        obs, reward, term, trunc, info = env.step(action)
        profit += reward
        done = term or trunc
        rows.append((info["t"], info["timestamp"], info["dam_position_kw"], info["id_adjustment_kw"],
                     info["pv_fcst_kw"], info["pv_kw"], info["p_batt_real_kw"], info["metered_kw"],
                     info["cap_up_commit_kw"], info["cap_down_commit_kw"], info["price_id_eur_mwh"],
                     info["imbalance_kwh"], info["id_eur"]))
    df = pd.DataFrame(rows, columns=["t", "timestamp", "x_kw", "y_kw", "pv_fcst_kw", "pv_kw", "p_batt_kw", "metered_kw",
                                     "r_up_kw", "r_dn_kw", "price_id_settle", "imbalance_kwh", "id_eur"])
    df["day"], df["activation_seed"] = int(day), int(seed)
    df["settle_auction"] = settle_auction
    df["seen_label"] = seen_label
    df["seen_value"] = seen_value
    df["decided_at"] = decided_at
    plan = plan_box.get("plan")
    df["d1_plan_battery_kw"] = np.asarray(plan["battery_kw"], dtype=float) if plan is not None and "battery_kw" in plan else np.nan
    return df, profit


def _capture_plan(agent):
    """Record the D-1 plan (incl. the QP's planned battery power) without changing it."""
    box = {}
    target = getattr(agent, "_planner", None)
    owner = target._agent if target is not None else agent
    original = owner._build_d1_plan

    def wrapped(obs):
        plan = original(obs)
        box["plan"] = plan
        return plan

    owner._build_d1_plan = wrapped
    return box


def job(controller: str, rl_seed: int, days: list[int], chunk: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    task, proto = Task.load(TASK), load_protocol()
    env = make_task_env(task, c_thr_eur_mwh=task.c_thr_eur_mwh)
    soc = float(proto["episode"]["soc_init"])
    if rl_seed < 0:
        agent = build_registered_algorithm(controller, env)
    else:
        from pv_bess_gym.rl.checkpoints import load_predict
        from pv_bess_gym.rl.wrappers import RLPolicyAgent
        sel = pd.read_csv(results_root() / "rl_training" / "selections.csv")
        row = sel[(sel.task == TASK) & (sel.controller == controller) & (sel.rl_seed == rl_seed)].iloc[0]
        run_dir = results_root() / "rl_training" / "runs" / "main" / TASK / controller / f"lr{row.learning_rate:g}_seed{rl_seed}"
        if not run_dir.exists():
            run_dir = results_root() / "rl_training" / "runs" / "main" / TASK / controller / f"lr{row.learning_rate}_seed{rl_seed}"
        from pv_bess_gym.rl.checkpoints import list_checkpoints
        ckpt = dict((int(s), p) for s, p in list_checkpoints(run_dir))[int(row.selected_steps)]
        agent = RLPolicyAgent(env, load_predict(run_dir, ckpt, env))
    box = _capture_plan(agent)
    frames, profits = [], []
    for day in days:
        for seed in proto["episode"]["activation_seeds"]:
            df, profit = _run(env, agent, day, seed, soc, box)
            df["controller"], df["rl_seed"] = controller, rl_seed
            frames.append(df)
            profits.append({"controller": controller, "rl_seed": rl_seed, "day": day, "activation_seed": seed, "profit_eur": profit})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(frames).to_csv(OUT / f"{controller}_rl{rl_seed}_{chunk:03d}.csv.gz", index=False)
    return profits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["baselines", "rl"])
    ap.add_argument("--controllers", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=13)
    args = ap.parse_args()
    proto = load_protocol()
    task = Task.load(TASK)
    probe = make_task_env(task, c_thr_eur_mwh=task.c_thr_eur_mwh)
    days = split_days(probe, "test", proto)
    chunks = [days[i:i + 8] for i in range(0, len(days), 8)]
    if args.kind == "baselines":
        units = [(c, -1) for c in (args.controllers or proto["baselines"])]
    else:
        units = [(c, s) for c in (args.controllers or proto["rl_baselines"]) for s in proto["rl_seeds"]]
    out = Parallel(n_jobs=args.jobs, verbose=5)(delayed(job)(c, s, d, i) for c, s in units for i, d in enumerate(chunks))
    prof = pd.DataFrame([r for part in out for r in part])
    tag = "_".join(sorted({c for c, _ in units}))
    prof.to_csv(OUT / f"profits_{tag}.csv", index=False)
    bench = pd.read_csv(results_root() / f"benchmark_{TASK}.csv")
    bench = bench[bench.split == "test"][["controller", "rl_seed", "day", "activation_seed", "profit_eur"]]
    m = prof.merge(bench, on=["controller", "rl_seed", "day", "activation_seed"], suffixes=("", "_bench"))
    print(f"episodes {len(prof)}, matched {len(m)}, max |profit - benchmark| = {np.abs(m.profit_eur - m.profit_eur_bench).max():.3e} EUR")


if __name__ == "__main__":
    main()

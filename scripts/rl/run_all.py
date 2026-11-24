"""Driver for the RL experiments (C7): tuning grid, main runs, checkpoint selection.

    python scripts/rl/run_all.py tune     # learning-rate grid on T1-Nominal, one seed
    python scripts/rl/run_all.py main --tasks T1-Nominal T2-Forecast T3-Gate

Each training and evaluation is a separate process in the Python environment
of its library (``--python`` for Stable-Baselines3, ``--python-omnisafe`` for
OmniSafe). Finished runs are skipped. The PPO-Lagrangian cost budget of a task
is the mean episode cost of DA-OPT-MPC on its validation split, read from
``results/benchmark_<task>.csv``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import load_protocol, results_root  # noqa: E402

ALGOS = ["ppo", "sac", "ppol"]
RUNS = results_root() / "rl_training" / "runs"


def cost_budget(task: str) -> float:
    """Mean episode cost of DA-OPT-MPC on the validation split (read from the per-job files,
    so RL can start before the oracle stages have been aggregated)."""
    files = sorted((results_root() / "parts" / task / "da-opt-mpc").glob("validation_*.csv"))
    if not files:
        raise SystemExit(f"{task}: run the baselines stage first (DA-OPT-MPC validation episodes are the PPOL cost budget).")
    rows = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return float(rows.cost_kw.mean())


def run_dir(tag: str, task: str, algo: str, lr: float, seed: int) -> Path:
    return RUNS / tag / task / f"da-opt-{algo}" / f"lr{lr:g}_seed{seed}"


def train_and_eval(args, tag: str, task: str, algo: str, lr: float, seed: int) -> str:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    target = run_dir(tag, task, algo, lr, seed)
    if algo == "ppol":
        python = args.python_omnisafe
        train = [python, "-W", "ignore", "scripts/rl/train_ppol.py", "--task", task, "--seed", str(seed), "--lr", str(lr),
                 "--cost-limit", repr(cost_budget(task)), "--tag", tag]
    else:
        python = args.python
        train = [python, "-W", "ignore", "scripts/rl/train_sb3.py", "--task", task, "--algo", algo, "--seed", str(seed),
                 "--lr", str(lr), "--tag", tag]
    extra = ["--c-thr", str(args.c_thr)] if args.c_thr is not None else []
    train += extra
    log = target.parent / f"{target.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        for command in (train, [python, "-W", "ignore", "scripts/rl/eval_rl.py", "--run-dir", str(target), *extra]):
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            if code:
                return f"FAILED ({code}): {' '.join(command[3:])}"
    return f"ok: {target.relative_to(ROOT)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["tune", "main"])
    parser.add_argument("--tasks", nargs="+", default=["T1-Nominal", "T2-Forecast", "T3-Gate"])
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--python-omnisafe", default=str(ROOT / ".venv-omnisafe/bin/python"))
    parser.add_argument("--c-thr", type=float, default=None, help="EUR/MWh, for T4/T5 until it is in the task files")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    proto = load_protocol()
    rl = proto["rl"]
    started = time.time()
    tuning_path = results_root() / "rl_training" / "tuning.json"

    if args.stage == "tune":
        grid, seed, task = rl["tuning"]["learning_rate"], int(rl["tuning"]["seed"]), rl["tuning"]["task"]
        jobs = [("tune", task, algo, float(lr), seed) for algo in ALGOS for lr in grid]
    else:
        selected = json.loads(tuning_path.read_text())["selected_learning_rate"]
        jobs = [("main", task, algo, float(selected[f"da-opt-{algo}"]), int(seed))
                for task in args.tasks for algo in ALGOS for seed in proto["rl_seeds"]]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for message in pool.map(lambda job: train_and_eval(args, *job), jobs):
            print(message, flush=True)

    if args.stage == "tune":
        table, selected = [], {}
        for algo in ALGOS:
            best = None
            for lr in grid:
                sel = json.loads((run_dir("tune", task, algo, float(lr), seed) / "selection.json").read_text())
                row = {"controller": f"da-opt-{algo}", "learning_rate": float(lr), "validation_score": sel["validation_score"],
                       "selected_steps": sel["selected_steps"], "validation_cost_kw": sel["validation_cost_kw"]}
                table.append(row)
                if best is None or row["validation_score"] > best["validation_score"]:
                    best = row
            selected[f"da-opt-{algo}"] = best["learning_rate"]
        tuning_path.write_text(json.dumps({
            "grid": {"learning_rate": grid}, "task": task, "seed": seed, "budget_steps": rl["total_timesteps"],
            "selection_rule": "highest validation score of the best checkpoint; all other hyper-parameters at library defaults",
            "results": table, "selected_learning_rate": selected,
            "cost_budget_kw_per_episode": cost_budget(task),
            "meta": run_metadata(started=started, seed=seed),
        }, indent=2), encoding="utf-8")
        print(json.dumps(selected))
    print(f"stage {args.stage} finished in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

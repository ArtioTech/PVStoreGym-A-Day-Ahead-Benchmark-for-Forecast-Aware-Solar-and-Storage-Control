"""C8 - compute measurements -> results/compute.json.

    python scripts/measure_compute.py          # = make compute

Measures on one core: environment steps per second, D-1 optimisation,
intraday MPC and real-time MPC solve times (mean and 95th percentile), and
collects oracle solve times, RL training times and per-step decision times
from the saved results.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym import build_registered_algorithm  # noqa: E402
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, results_root, split_days  # noqa: E402

RES = results_root()


def stats(values, unit_scale: float = 1000.0) -> dict:
    v = np.asarray(values, dtype=float) * unit_scale
    return {"n": int(len(v)), "mean": float(v.mean()), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def sysctl(key: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    proto = load_protocol()
    env = make_task_env("T1-Nominal")
    days = split_days(env, "test", proto)[:20]

    # environment steps per second, single core, pre-sampled random actions
    env.action_space.seed(0)
    actions = [env.action_space.sample() for _ in range(96)]
    elapsed, steps = 0.0, 0
    for day in days:
        env.reset(options={"day": day, "soc_init": 0.5, "activation_seed": 0})
        t0 = time.perf_counter()
        for action in actions:
            env.step(action)
        elapsed += time.perf_counter() - t0
        steps += 96
    env_steps_per_second = steps / elapsed

    # solve times of the optimisation layers (DA-OPT-MPC uses all three)
    agent = build_registered_algorithm("da-opt-mpc", env)
    timers: dict[str, list[float]] = {"d1": [], "intraday_mpc": [], "realtime_mpc": []}

    def timed(name, func):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            timers[name].append(time.perf_counter() - t0)
            return result
        return wrapper

    agent._build_d1_plan = timed("d1", agent._build_d1_plan)
    agent._solve_intraday_mpc = timed("intraday_mpc", agent._solve_intraday_mpc)
    agent._solve_realtime_mpc = timed("realtime_mpc", agent._solve_realtime_mpc)
    for day in days:
        agent.reset()
        obs, _ = env.reset(options={"day": day, "soc_init": 0.5, "activation_seed": 0})
        done = False
        while not done:
            obs, _, term, trunc, _ = env.step(agent.act(obs))
            done = term or trunc

    oracle = pd.concat([pd.read_csv(p) for p in glob.glob(str(RES / "parts/oracle/*/*.csv"))])
    out = {
        "environment_steps_per_second_single_core": float(env_steps_per_second),
        "d1_optimisation_ms": {**stats(timers["d1"][1:]), "first_call_incl_compilation_ms": float(1000 * timers["d1"][0]),
                               "note": "convex QP, CLARABEL via CVXPY, compiled once and re-solved"},
        "intraday_mpc_ms": stats(timers["intraday_mpc"]),
        "realtime_mpc_ms": stats(timers["realtime_mpc"]),
        "oracle_milp_s": {**stats(oracle.solve_seconds, 1.0), "max_mip_gap": float(oracle.mip_gap.max()),
                          "note": "one delivery day, HiGHS via scipy.optimize.milp"},
        "rl_training": {}, "decision_time_ms_per_step": {}, "cpu_hours": {},
    }
    for path in sorted(glob.glob(str(RES / "rl_training/runs/main/*/*/*/train.json"))):
        meta = json.loads(Path(path).read_text())
        entry = out["rl_training"].setdefault(meta["controller"], {"train_seconds": [], "steps_per_second": [], "total_timesteps": meta["total_timesteps"]})
        entry["train_seconds"].append(meta["train_seconds"])
        entry["steps_per_second"].append(meta["steps_per_second"])
    for name, entry in out["rl_training"].items():
        entry.update({"runs": len(entry["train_seconds"]), "mean_train_minutes_per_seed": float(np.mean(entry["train_seconds"]) / 60.0),
                      "mean_steps_per_second": float(np.mean(entry["steps_per_second"]))})
        del entry["train_seconds"], entry["steps_per_second"]
    bench = RES / "benchmark_T1-Nominal.csv"
    if bench.exists():
        df = pd.read_csv(bench)
        out["decision_time_ms_per_step"] = {c: float(1000 * g.decision_seconds_per_step.mean()) for c, g in df.groupby("controller")}
    seconds: dict[str, float] = {}
    for path in glob.glob(str(RES / "parts/**/*.meta.json"), recursive=True):
        meta = json.loads(Path(path).read_text())
        seconds[meta.get("stage", "other")] = seconds.get(meta.get("stage", "other"), 0.0) + meta.get("wall_clock_seconds", 0.0)
    for path in glob.glob(str(RES / "rl_training/runs/*/*/*/*/train.json")):
        seconds["rl-train"] = seconds.get("rl-train", 0.0) + json.loads(Path(path).read_text())["train_seconds"]
    out["cpu_hours"] = {k: v / 3600.0 for k, v in seconds.items()}
    out["cpu_hours"]["total"] = sum(seconds.values()) / 3600.0
    out["hardware"] = {"cpu": sysctl("machdep.cpu.brand_string"), "cores": os.cpu_count(),
                       "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30) if sysctl("hw.memsize").isdigit() else None,
                       "platform": platform.platform(), "accelerator": "none (CPU only)"}
    out["meta"] = run_metadata(started=started, command="make compute", seed=0)
    path = RES / "compute.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1)[:2500])
    print(f"-> {path}")


if __name__ == "__main__":
    main()

"""Journal benchmark runner (configs/journal.yaml). One stage per result family.

    python scripts/run_journal.py oracle     --tasks T1-Nominal T3-Gate
    python scripts/run_journal.py baselines  --tasks T1-Nominal T2-Forecast T3-Gate
    python scripts/run_journal.py aggregate  --tasks T1-Nominal T2-Forecast T3-Gate

Every unit of work writes one CSV under ``results/parts/`` and is skipped when
that file exists (``--force`` recomputes). ``aggregate`` joins them into
``results/benchmark_<task>.csv``.

Evaluation sets: ``test`` and ``validation`` of the protocol, and ``heldout``
(the within-regime weeks). Every (day, activation seed) pair is one episode
from SOC 0.5. The oracle does not depend on what the controllers may observe,
so tasks that differ only in information (T1/T2) share one oracle run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym import build_registered_algorithm  # noqa: E402
from pv_bess_gym.benchmark.episodes import run_episode  # noqa: E402
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import Task, load_protocol, make_task_env, results_root, split_days  # noqa: E402

EVAL_SPLITS = ["test", "validation", "heldout"]
RESULTS = results_root()          # results/ or results/<results_subdir> of the active protocol
PARTS = RESULTS / "parts"
CHUNK_DAYS = 8


def oracle_key(task: Task, c_thr: float) -> str:
    """Tasks with the same gate lead and degradation cost have the same oracle."""
    return f"gate{int(task.gate_closure_minutes)}_cthr{c_thr:g}"


def resolve_c_thr(task: Task, cli_value: float | None) -> float:
    value = task.c_thr_eur_mwh if cli_value is None else cli_value
    if value is None:
        raise SystemExit(f"{task.name} needs --c-thr (author value); it is not set in configs/tasks/{task.name}.yaml.")
    return float(value)


def chunks(days: list[int]) -> list[list[int]]:
    return [days[i:i + CHUNK_DAYS] for i in range(0, len(days), CHUNK_DAYS)]


def _write(rows: list[dict], path: Path, started: float, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(run_metadata(started=started, **meta)), encoding="utf-8")


def job_oracle(task_name: str, c_thr: float, split: str, chunk_id: int, days: list[int], force: bool) -> None:
    warnings.filterwarnings("ignore")
    from pv_bess_gym.oracle import replay_oracle, solve_oracle

    task, proto = Task.load(task_name), load_protocol()
    path = PARTS / "oracle" / oracle_key(task, c_thr) / f"{split}_{chunk_id:03d}.csv"
    if path.exists() and not force:
        return
    started = time.time()
    env = make_task_env(task, c_thr_eur_mwh=c_thr)
    soc = float(proto["episode"]["soc_init"])
    rows = []
    for day in days:
        for seed in proto["episode"]["activation_seeds"]:
            res = solve_oracle(env, day, soc_init=soc, activation_seed=seed,
                               time_limit_s=float(proto["oracle"]["time_limit_s"]),
                               mip_rel_gap=float(proto["oracle"]["mip_rel_gap"]))
            rows.append({
                "split": split, "day": day, "date": env.day_date(day), "activation_seed": seed,
                "U": res.upper_bound_eur, "U_incumbent": res.objective_eur, "mip_gap": res.mip_gap,
                "U_rep": replay_oracle(env, res, soc_init=soc), "solve_seconds": res.solve_seconds,
                "retries": res.retries,
                **{f"oracle_{k}": v for k, v in res.breakdown.items()},
            })
    _write(rows, path, started, stage="oracle", oracle_key=oracle_key(task, c_thr), split=split,
           seed=list(proto["episode"]["activation_seeds"]))


def job_oracle_deliver(task_name: str, c_thr: float, split: str, chunk_id: int, days: list[int], force: bool) -> None:
    """ORACLE-DELIVER (R6): the oracle constrained to meter its own schedule."""
    warnings.filterwarnings("ignore")
    from pv_bess_gym.oracle import solve_oracle

    task, proto = Task.load(task_name), load_protocol()
    eps = float(proto["oracle"]["deliver_epsilon_kwh"])
    path = PARTS / "oracle_deliver" / oracle_key(task, c_thr) / f"{split}_{chunk_id:03d}.csv"
    if path.exists() and not force:
        return
    started = time.time()
    env = make_task_env(task, c_thr_eur_mwh=c_thr)
    soc = float(proto["episode"]["soc_init"])
    rows = []
    for day in days:
        for seed in proto["episode"]["activation_seeds"]:
            res = solve_oracle(env, day, soc_init=soc, activation_seed=seed, allow_imbalance=False,
                               deliver_epsilon_kwh=eps,
                               time_limit_s=float(proto["oracle"]["time_limit_s"]),
                               mip_rel_gap=float(proto["oracle"]["mip_rel_gap"]))
            rows.append({
                "split": split, "day": day, "date": env.day_date(day), "activation_seed": seed,
                "U_D": res.upper_bound_eur, "U_D_incumbent": res.objective_eur, "mip_gap_D": res.mip_gap,
                "solve_seconds_D": res.solve_seconds, "retries_D": res.retries,
                **{f"oracle_D_{k}": v for k, v in res.breakdown.items()},
            })
    _write(rows, path, started, stage="oracle_deliver", oracle_key=oracle_key(task, c_thr), split=split,
           deliver_epsilon_kwh=eps, seed=list(proto["episode"]["activation_seeds"]))


def job_baseline(task_name: str, c_thr: float, controller: str, split: str, chunk_id: int, days: list[int], force: bool) -> None:
    warnings.filterwarnings("ignore")
    task, proto = Task.load(task_name), load_protocol()
    path = PARTS / task_name / controller / f"{split}_{chunk_id:03d}.csv"
    if path.exists() and not force:
        return
    started = time.time()
    env = make_task_env(task, c_thr_eur_mwh=c_thr)
    agent = build_registered_algorithm(controller, env)
    soc = float(proto["episode"]["soc_init"])
    rows = []
    for day in days:
        for seed in proto["episode"]["activation_seeds"]:
            row = run_episode(env, agent, day, seed, soc)
            rows.append({"task": task_name, "controller": controller, "split": split, "rl_seed": -1, **row})
    _write(rows, path, started, stage="baselines", task=task_name, controller=controller, split=split,
           seed=list(proto["episode"]["activation_seeds"]))


def aggregate(task_name: str, c_thr: float) -> Path:
    task, proto = Task.load(task_name), load_protocol()
    okey = oracle_key(task, c_thr)
    oracle = pd.concat([pd.read_csv(p) for p in sorted((PARTS / "oracle" / okey).glob("*.csv"))], ignore_index=True)
    runs = pd.concat([pd.read_csv(p) for p in sorted((PARTS / task_name).rglob("*.csv"))], ignore_index=True)
    keys = ["split", "day", "activation_seed"]
    anchors = runs[runs.controller == "da-only"][keys + ["profit_eur"]].rename(columns={"profit_eur": "J_L"})
    df = runs.merge(oracle[keys + ["U", "U_incumbent", "U_rep", "mip_gap"]], on=keys, how="left").merge(anchors, on=keys, how="left")
    if df[["U", "J_L"]].isna().any().any():
        raise RuntimeError(f"{task_name}: episodes without an oracle bound or a DA-ONLY reference.")
    violations = df[df.profit_eur > df.U + 1e-6]
    if len(violations):
        raise AssertionError(f"{task_name}: {len(violations)} episode(s) with profit above the oracle bound:\n"
                             f"{violations[['controller', 'split', 'date', 'activation_seed', 'profit_eur', 'U']].head()}")
    rng_eur = df.U - df.J_L
    df["score_valid"] = rng_eur >= float(proto["score"]["min_range_eur"])
    df["score"] = np.where(df.score_valid, (df.profit_eur - df.J_L) / rng_eur.where(df.score_valid, np.nan), np.nan)
    deliver_checks = {}
    d_tasks = proto["oracle"].get("deliver_tasks")          # rev1: ORACLE-DELIVER only for these tasks
    if "deliver_epsilon_kwh" in proto["oracle"] and (d_tasks is None or task_name in d_tasks):
        dfiles = sorted((PARTS / "oracle_deliver" / okey).glob("*.csv"))
        if not dfiles:
            raise RuntimeError(f"{task_name}: no ORACLE-DELIVER results; run the oracle_deliver stage.")
        od = pd.concat([pd.read_csv(p) for p in dfiles], ignore_index=True)
        oracle = oracle.merge(od, on=["split", "day", "date", "activation_seed"], how="left")
        df = df.merge(od[keys + ["U_D", "U_D_incumbent", "mip_gap_D"]], on=keys, how="left")
        d_splits = list(proto["oracle"].get("deliver_splits", EVAL_SPLITS))
        if df[df.split.isin(d_splits)].U_D.isna().any():
            raise RuntimeError(f"{task_name}: episodes without an ORACLE-DELIVER bound in splits {d_splits}.")
        df.loc[~df.split.isin(d_splits), ["U_D", "U_D_incumbent", "mip_gap_D"]] = np.nan
        oracle.loc[~oracle.split.isin(d_splits), [c for c in od.columns if c not in keys + ["date"]]] = np.nan
        # ORACLE-DELIVER <= ORACLE (its feasible set is a subset; compare incumbents
        # with the bound, and bounds allowing for the MIP gaps of both solves)
        above = oracle[oracle.U_D_incumbent > oracle.U + 1e-6]
        if len(above):
            raise AssertionError(f"{task_name}: {len(above)} episode(s) with ORACLE-DELIVER above ORACLE.")
        ref = df[df.controller == "da-only"]
        below_jl = ref[ref.U_D < ref.J_L - 1e-6]
        rng_d = df.U_D - df.J_L
        df["score_D_valid"] = rng_d >= float(proto["score"]["min_range_eur"])
        df["score_D"] = np.where(df.score_D_valid, (df.profit_eur - df.J_L) / rng_d.where(df.score_D_valid, np.nan), np.nan)
        deliver_checks = {
            "deliver_epsilon_kwh": float(proto["oracle"]["deliver_epsilon_kwh"]),
            "oracle_deliver_above_oracle": 0,
            "episodes_U_D_below_J_L": int(len(below_jl)),
            "controller_episodes_above_U_D": int((df.profit_eur > df.U_D + 1e-6).sum()),
        }
        if len(below_jl):
            print(f"WARNING {task_name}: {len(below_jl)} episode(s) with ORACLE-DELIVER below DA-ONLY (DA-ONLY earns imbalance revenue there)")
    # per-day oracle results (bound, replay, components, solve time), kept next to the benchmark file
    oracle.sort_values(keys).to_csv(RESULTS / f"oracle_{task_name}.csv", index=False)
    out = RESULTS / f"benchmark_{task_name}.csv"
    df.sort_values(["split", "controller", "rl_seed", "day", "activation_seed"]).to_csv(out, index=False)
    parts_meta = [json.loads(p.read_text()) for p in sorted((PARTS / task_name).rglob("*.meta.json"))]
    parts_meta += [json.loads(p.read_text()) for p in sorted((PARTS / "oracle" / okey).glob("*.meta.json"))]
    parts_meta += [json.loads(p.read_text()) for p in sorted((PARTS / "oracle_deliver" / okey).glob("*.meta.json"))]
    meta = run_metadata(task=task_name, c_thr_eur_mwh=c_thr, oracle_key=okey,
                        seed={"activation_seeds": proto["episode"]["activation_seeds"]},
                        source_commits=sorted({m["commit"] for m in parts_meta}),
                        source_dirty=any(m["dirty"] for m in parts_meta),
                        wall_clock_seconds=float(sum(m.get("wall_clock_seconds", 0.0) for m in parts_meta)),
                        n_episodes=int(len(df)), oracle_dominance_violations=0, protocol=proto["name"], **deliver_checks)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{task_name}: {len(df)} episodes, {df.controller.nunique()} controllers -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["oracle", "oracle_deliver", "baselines", "aggregate"])
    parser.add_argument("--tasks", nargs="+", default=["T1-Nominal", "T2-Forecast", "T3-Gate"])
    parser.add_argument("--splits", nargs="+", default=EVAL_SPLITS, choices=EVAL_SPLITS)
    parser.add_argument("--controllers", nargs="+", default=None)
    parser.add_argument("--c-thr", type=float, default=None, help="EUR/MWh, for T4/T5 until it is in the task files")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    proto = load_protocol()
    tasks = [Task.load(name) for name in args.tasks]
    c_thr = {task.name: resolve_c_thr(task, args.c_thr) for task in tasks}
    probe = make_task_env(tasks[0], c_thr_eur_mwh=c_thr[tasks[0].name])
    day_chunks = {split: chunks(split_days(probe, split, proto)) for split in args.splits}
    run = Parallel(n_jobs=args.jobs, verbose=5)
    started = time.time()

    if args.stage in ("oracle", "oracle_deliver"):
        job = job_oracle if args.stage == "oracle" else job_oracle_deliver
        seen, jobs = set(), []
        for task in tasks:
            key = oracle_key(task, c_thr[task.name])
            if key in seen:
                continue
            seen.add(key)
            splits = args.splits
            if args.stage == "oracle_deliver":   # only the splits that report S_D (protocol oracle.deliver_splits)
                splits = [s for s in splits if s in proto["oracle"].get("deliver_splits", EVAL_SPLITS)]
                d_tasks = proto["oracle"].get("deliver_tasks")   # and only the tasks listed (rev1)
                if d_tasks is not None and task.name not in d_tasks:
                    continue
            jobs += [delayed(job)(task.name, c_thr[task.name], s, i, d, args.force)
                     for s in splits for i, d in enumerate(day_chunks[s])]
        run(jobs)
    elif args.stage == "baselines":
        controllers = args.controllers or proto["baselines"]
        run(delayed(job_baseline)(task.name, c_thr[task.name], ctrl, s, i, d, args.force)
            for task in tasks for ctrl in controllers for s in args.splits for i, d in enumerate(day_chunks[s]))
    else:
        for task in tasks:
            aggregate(task.name, c_thr[task.name])
    print(f"stage {args.stage} finished in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()

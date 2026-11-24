"""Aggregates with confidence intervals -> results/summary_<task>.json (C2-C6).

Run with the Python environment that has ``rliable`` (the OmniSafe environment):

    python scripts/summarise_journal.py --tasks T1-Nominal T2-Forecast T3-Gate

Statistics
----------
- Normalised score of an episode: ``S = (J - J_L) / (U - J_L)``; episodes with
  ``U - J_L`` below ``score.min_range_eur`` are excluded and counted.
- Task score: interquartile mean (IQM) over episodes (days x activation seeds,
  and training seeds for RL) with a 95% stratified bootstrap CI from
  ``rliable`` (one stratum, the task); episode mean and episode median with
  the same bootstrap.
- Daily-profit means and the layer attribution use a 95% percentile bootstrap
  over delivery days (a day's activation seeds are resampled together).
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rliable import library as rly  # noqa: E402
from rliable import metrics  # noqa: E402

from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import load_protocol, results_root  # noqa: E402

REPS = 5000
ORDER = ["da-only", "da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc", "da-opt-ppo", "da-opt-sac", "da-opt-ppol"]
COMPONENTS = ["dam_eur", "id_eur", "afrr_capacity_eur", "afrr_activation_eur", "imbalance_eur", "degradation_eur", "penalties_eur"]


def day_bootstrap(frame: pd.DataFrame, column: str, reps: int = REPS, seed: int = 0) -> dict:
    """Mean of ``column`` with a 95% CI from resampling delivery days.

    Every day has the same number of episodes, so the overall mean equals the
    mean of the per-day means; days (with all their episodes) are resampled.
    """
    per_day = frame.groupby("day")[column].mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = per_day[rng.integers(len(per_day), size=(reps, len(per_day)))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": float(per_day.mean()), "ci95": [float(lo), float(hi)]}


def score_aggregates(scores: np.ndarray, seed: int = 0) -> dict:
    """IQM, mean and median with rliable's stratified bootstrap (runs x 1 task)."""
    matrix = np.asarray(scores, dtype=float).reshape(-1, 1)
    # rliable's aggregate_median is the median across *tasks* of per-task means;
    # with one task it equals the mean, so the episode median is taken directly.
    func = lambda x: np.array([metrics.aggregate_iqm(x), float(np.mean(x)), float(np.median(x))])  # noqa: E731
    point, interval = rly.get_interval_estimates({"c": matrix}, func, reps=REPS, random_state=np.random.RandomState(seed))
    names = ["iqm", "mean", "median"]
    return {name: {"value": float(point["c"][i]), "ci95": [float(interval["c"][0][i]), float(interval["c"][1][i])]}
            for i, name in enumerate(names)}


def cvar(values: np.ndarray, alpha: float = 0.05) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    k = max(1, int(np.ceil(alpha * len(ordered))))
    return float(ordered[:k].mean())


def pooled_correlation(frame: pd.DataFrame) -> float:
    n, sx, sy = frame.xy_n.sum(), frame.xy_sx.sum(), frame.xy_sy.sum()
    sxx, syy, sxy = frame.xy_sxx.sum(), frame.xy_syy.sum(), frame.xy_sxy.sum()
    cov = sxy - sx * sy / n
    var = (sxx - sx * sx / n) * (syy - sy * sy / n)
    return float(cov / np.sqrt(var)) if var > 0 else float("nan")


def summarise(task: str) -> dict:
    proto = load_protocol()
    df = pd.read_csv(results_root() / f"benchmark_{task}.csv")
    source = json.loads((results_root() / f"benchmark_{task}.meta.json").read_text())
    controllers = [c for c in ORDER if c in set(df.controller)]
    out: dict = {"task": task, "splits": {}}
    for split in ["test", "validation", "heldout"]:
        part = df[df.split == split]
        if part.empty:
            continue
        ref = part[part.controller == "da-only"]
        per_split: dict = {
            "n_days": int(ref.day.nunique()), "n_activation_seeds": int(ref.activation_seed.nunique()),
            "date_range": [str(ref.date.min()), str(ref.date.max())],
            "anchors": {
                "U_mean_daily_eur": day_bootstrap(ref, "U"),
                "U_rep_mean_daily_eur": day_bootstrap(ref, "U_rep"),
                "J_L_mean_daily_eur": day_bootstrap(ref, "J_L"),
                "relative_gap_U_minus_U_rep_over_U": float((ref.U.sum() - ref.U_rep.sum()) / ref.U.sum()),
                "max_mip_gap": float(ref.mip_gap.max()),
                "episodes_excluded_from_score": int((~ref.score_valid).sum()),
                "days_with_an_excluded_episode": int(ref[~ref.score_valid].day.nunique()),
            },
            "controllers": {},
        }
        has_d = "U_D" in part.columns and part.U_D.notna().all()   # S_D only on the ORACLE-DELIVER splits
        if has_d:   # R6: delivery-constrained oracle and the second score S_D
            per_split["anchors"].update({
                "U_D_mean_daily_eur": day_bootstrap(ref, "U_D"),
                "max_mip_gap_D": float(ref.mip_gap_D.max()),
                "episodes_excluded_from_score_D": int((~ref.score_D_valid).sum()),
                "episodes_U_D_below_J_L": int((ref.U_D < ref.J_L - 1e-6).sum()),
            })
        for controller in controllers:
            rows = part[part.controller == controller]
            valid = rows[rows.score_valid]
            entry = {
                "n_episodes": int(len(rows)), "n_rl_seeds": int(rows.rl_seed.nunique()) if (rows.rl_seed >= 0).any() else 0,
                "score": score_aggregates(valid.score.to_numpy()),
                "mean_daily_profit_eur": day_bootstrap(rows, "profit_eur"),
                "cvar5_daily_profit_eur": cvar(rows.profit_eur.to_numpy()),
                "reserve_deliverability": float(rows.activation_delivered_kwh.sum() / rows.activation_requested_kwh.sum())
                if rows.activation_requested_kwh.sum() > 0 else None,
                "penalties_eur_per_day": float(rows.penalties_eur.mean()),
                "cost_kw_per_day": float(rows.cost_kw.mean()),
                "equivalent_full_cycles_per_day": float(rows.equivalent_full_cycles.mean()),
                "curtailment_kwh_per_day": float(rows.curtailed_kwh.mean()),
                "decision_time_ms_per_step": float(1000.0 * rows.decision_seconds_per_step.mean()),
                "components_eur_per_day": {c: float(rows[c].mean()) for c in COMPONENTS},
                "imbalance": {
                    "mean_sign_of_step_imbalance": float(rows.imbalance_sign_mean.mean()),
                    "mean_imbalance_kwh_per_day": float(rows.imbalance_kwh.mean()),
                    "correlation_with_id_minus_imbalance_price": pooled_correlation(rows),
                },
            }
            if "scheduled_kwh" in rows.columns:
                steps = 96.0
                entry["delivery"] = {
                    "scheduled_kwh_per_day": float(rows.scheduled_kwh.mean()),
                    "metered_kwh_per_day": float(rows.metered_kwh.mean()),
                    "share_steps_short": float(rows.short_imbalance_steps.sum() / (steps * len(rows))),
                    "share_steps_long": float(rows.long_imbalance_steps.sum() / (steps * len(rows))),
                }
            if has_d:
                valid_d = rows[rows.score_D_valid]
                entry["score_D"] = score_aggregates(valid_d.score_D.to_numpy())
            if entry["n_rl_seeds"]:
                per_seed = valid.groupby("rl_seed").score.agg(lambda s: metrics.aggregate_iqm(s.to_numpy().reshape(-1, 1)))
                entry["score_iqm_per_rl_seed"] = {int(k): float(v) for k, v in per_seed.items()}
                entry["score_iqm_std_over_rl_seeds"] = float(per_seed.std(ddof=1))
                if has_d:
                    per_seed_d = valid_d.groupby("rl_seed").score_D.agg(lambda s: metrics.aggregate_iqm(s.to_numpy().reshape(-1, 1)))
                    entry["score_D_iqm_std_over_rl_seeds"] = float(per_seed_d.std(ddof=1))
            per_split["controllers"][controller] = entry

        # C5 layer attribution on paired (day, seed) episodes
        need = ["da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc"]
        if all(c in controllers for c in need):
            wide = part[part.controller.isin(need) & part.score_valid].pivot_table(
                index=["day", "activation_seed"], columns="controller", values="score").dropna().reset_index()
            wide["d_id"] = 0.5 * ((wide["da-opt-id-mpc"] - wide["da-opt-rbc"]) + (wide["da-opt-mpc"] - wide["da-opt-rt-mpc"]))
            wide["d_rt"] = 0.5 * ((wide["da-opt-rt-mpc"] - wide["da-opt-rbc"]) + (wide["da-opt-mpc"] - wide["da-opt-id-mpc"]))
            wide["inter"] = wide["da-opt-mpc"] - wide["da-opt-id-mpc"] - wide["da-opt-rt-mpc"] + wide["da-opt-rbc"]
            per_split["layer_attribution"] = {
                "n_paired_episodes": int(len(wide)),
                "delta_id": day_bootstrap(wide, "d_id"),
                "delta_rt": day_bootstrap(wide, "d_rt"),
                "interaction": day_bootstrap(wide, "inter"),
            }
            if has_d:
                wd = part[part.controller.isin(need) & part.score_D_valid].pivot_table(
                    index=["day", "activation_seed"], columns="controller", values="score_D").dropna().reset_index()
                wd["d_id"] = 0.5 * ((wd["da-opt-id-mpc"] - wd["da-opt-rbc"]) + (wd["da-opt-mpc"] - wd["da-opt-rt-mpc"]))
                wd["d_rt"] = 0.5 * ((wd["da-opt-rt-mpc"] - wd["da-opt-rbc"]) + (wd["da-opt-mpc"] - wd["da-opt-id-mpc"]))
                wd["inter"] = wd["da-opt-mpc"] - wd["da-opt-id-mpc"] - wd["da-opt-rt-mpc"] + wd["da-opt-rbc"]
                per_split["layer_attribution_D"] = {
                    "n_paired_episodes": int(len(wd)),
                    "delta_id": day_bootstrap(wd, "d_id"), "delta_rt": day_bootstrap(wd, "d_rt"),
                    "interaction": day_bootstrap(wd, "inter"),
                }
        out["splits"][split] = per_split
    out["meta"] = run_metadata(task=task, bootstrap_reps=REPS, seed=0, source_commits=source["source_commits"],
                               source_wall_clock_seconds=source["wall_clock_seconds"],
                               score_min_range_eur=proto["score"]["min_range_eur"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=["T1-Nominal", "T2-Forecast", "T3-Gate"])
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    for task in args.tasks:
        started = time.time()
        summary = summarise(task)
        summary["meta"]["wall_clock_seconds"] = time.time() - started
        path = results_root() / f"summary_{task}.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        test = summary["splits"]["test"]
        print(f"{task} test: U {test['anchors']['U_mean_daily_eur']['value']:.1f}  U_rep {test['anchors']['U_rep_mean_daily_eur']['value']:.1f}  "
              f"J_L {test['anchors']['J_L_mean_daily_eur']['value']:.1f}")
        for name, entry in test["controllers"].items():
            iqm = entry["score"]["iqm"]
            print(f"  {name:14s} IQM {iqm['value']:+.4f} [{iqm['ci95'][0]:+.4f}, {iqm['ci95'][1]:+.4f}]  profit {entry['mean_daily_profit_eur']['value']:.1f}")
        print(f"-> {path}")


if __name__ == "__main__":
    main()

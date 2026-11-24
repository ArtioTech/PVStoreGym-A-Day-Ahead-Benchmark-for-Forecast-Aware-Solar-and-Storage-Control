"""Collect the RL training records into results/rl_training/ (C7).

    python scripts/rl/collect_rl.py

Writes ``curves.csv`` (validation score and cost of every checkpoint of every
main run), ``selections.csv`` (selected checkpoint per run) and
``hyperparameters.json`` (budget, learning rate, network, discount, tuning
grid, selection rule, cost budgets). Checkpoints themselves stay under
``results/rl_training/runs/`` and are not version-controlled.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import load_protocol, results_root  # noqa: E402

OUT = results_root() / "rl_training"
curves, selections, hyper = [], [], {}
for train_path in sorted((OUT / "runs" / "main").glob("*/*/*/train.json")):
    run = train_path.parent
    meta = json.loads(train_path.read_text())
    sel = json.loads((run / "selection.json").read_text())
    curve = pd.read_csv(run / "validation_curve.csv")
    curve.insert(0, "rl_seed", meta["seed"])
    curve.insert(0, "controller", meta["controller"])
    curve.insert(0, "task", meta["task"])
    curves.append(curve)
    selections.append({
        "task": meta["task"], "controller": meta["controller"], "rl_seed": meta["seed"],
        "learning_rate": meta["learning_rate"], "selected_steps": sel["selected_steps"],
        "selected_checkpoint": sel["selected_checkpoint"], "validation_score": sel["validation_score"],
        "validation_cost_kw": sel["validation_cost_kw"], "train_seconds": meta["train_seconds"],
        "steps_per_second": meta["steps_per_second"], "cost_limit": meta.get("cost_limit"),
        "commit": meta["meta"]["commit"],
    })
    hyper.setdefault(meta["controller"], {
        "library": meta["library"], "algorithm": meta["algo"], "budget_steps": meta["total_timesteps"],
        "learning_rate": meta["learning_rate"], "network": meta["network"], "discount": meta["discount"],
        "checkpoint_every_steps": meta["checkpoint_every"], "other": meta["hyperparameters"],
    })
    if meta.get("cost_limit") is not None:
        hyper[meta["controller"]].setdefault("cost_budget_kw_per_episode", {})[meta["task"]] = meta["cost_limit"]
proto = load_protocol()
pd.concat(curves, ignore_index=True).to_csv(OUT / "curves.csv", index=False)
pd.DataFrame(selections).to_csv(OUT / "selections.csv", index=False)
tuning = json.loads((OUT / "tuning.json").read_text())
(OUT / "hyperparameters.json").write_text(json.dumps({
    "controllers": hyper,
    "seeds": proto["rl_seeds"],
    "tuning": {k: tuning[k] for k in ("grid", "task", "seed", "selection_rule", "results", "selected_learning_rate")},
    "checkpoint_selection": "checkpoint with the highest mean normalised score on the validation split "
                            f"(activation seeds {proto['rl']['selection_activation_seeds']})",
    "training_episodes": "train split without the held-out weeks, SOC 0.5, activation seeds "
                         f"{proto['rl']['train_activation_seeds'][0]}-{proto['rl']['train_activation_seeds'][-1]}",
    "meta": run_metadata(),
}, indent=2), encoding="utf-8")
print(f"{len(selections)} runs -> {OUT / 'curves.csv'}, selections.csv, hyperparameters.json")

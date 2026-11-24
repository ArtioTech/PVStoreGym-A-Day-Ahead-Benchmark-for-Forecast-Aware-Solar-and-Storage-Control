"""Score the checkpoints of an RL run on the validation split, select one, evaluate it.

Run with the Python environment of the library that trained the run
(Stable-Baselines3 runs: main environment; OmniSafe runs: OmniSafe environment).

    python scripts/rl/eval_rl.py --run-dir results/rl_training/runs/main/T1-Nominal/da-opt-ppo/lr0.0003_seed0

1. Every checkpoint is evaluated on the validation days with the selection
   activation seeds; its validation score is the mean normalised score
   ``(J - J_L) / (U - J_L)`` over those episodes.
2. The checkpoint with the highest validation score is selected.
3. For runs tagged ``main`` the selected checkpoint is evaluated on the test,
   validation and held-out days with all activation seeds; the episodes go to
   ``results/parts/<task>/<controller>/rlseed<k>_<split>.csv``.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.episodes import run_episode  # noqa: E402
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import Task, load_protocol, make_task_env, results_root, split_days  # noqa: E402
from pv_bess_gym.rl.checkpoints import list_checkpoints, load_predict  # noqa: E402
from pv_bess_gym.rl.wrappers import RLPolicyAgent  # noqa: E402

PARTS = results_root() / "parts"
KEYS = ["day", "activation_seed"]


def anchors(task: Task, c_thr: float, split: str) -> pd.DataFrame:
    key = f"gate{int(task.gate_closure_minutes)}_cthr{c_thr:g}"
    oracle = pd.concat([pd.read_csv(p) for p in sorted((PARTS / "oracle" / key).glob(f"{split}_*.csv"))])
    ref = pd.concat([pd.read_csv(p) for p in sorted((PARTS / task.name / "da-only").glob(f"{split}_*.csv"))])
    return oracle[KEYS + ["U"]].merge(ref[KEYS + ["profit_eur"]].rename(columns={"profit_eur": "J_L"}), on=KEYS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--c-thr", type=float, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    meta = json.loads((run_dir / "train.json").read_text(encoding="utf-8"))
    if (run_dir / "selection.json").exists() and not args.force:
        print(f"exists: {run_dir / 'selection.json'}")
        return
    proto = load_protocol()
    task = Task.load(meta["task"])
    c_thr = task.c_thr_eur_mwh if args.c_thr is None else args.c_thr
    env = make_task_env(task, c_thr_eur_mwh=c_thr)
    soc = float(proto["episode"]["soc_init"])
    min_range = float(proto["score"]["min_range_eur"])
    started = time.time()

    # 1. validation curve
    val = anchors(task, float(c_thr), "validation")
    val_days = split_days(env, "validation", proto)
    curve = []
    for steps, checkpoint in list_checkpoints(run_dir):
        agent = RLPolicyAgent(env, load_predict(run_dir, checkpoint, env))
        rows = pd.DataFrame([run_episode(env, agent, d, s, soc) for d in val_days for s in proto["rl"]["selection_activation_seeds"]])
        rows = rows.merge(val, on=KEYS)
        valid = (rows.U - rows.J_L) >= min_range
        score = ((rows.profit_eur - rows.J_L) / (rows.U - rows.J_L))[valid]
        curve.append({"steps": steps, "checkpoint": checkpoint.name, "validation_score": float(score.mean()),
                      "validation_profit_eur": float(rows.profit_eur.mean()),
                      "validation_cost_kw": float(rows.cost_kw.mean()), "n_episodes": int(len(rows))})
    curve = pd.DataFrame(curve)
    curve.to_csv(run_dir / "validation_curve.csv", index=False)

    # 2. selection
    best = curve.loc[curve.validation_score.idxmax()]
    selected = dict(zip(*[[c[0] for c in list_checkpoints(run_dir)], [c[1] for c in list_checkpoints(run_dir)]]))[int(best.steps)]

    # 3. final evaluation
    n_final = 0
    if meta["tag"] == "main":
        agent = RLPolicyAgent(env, load_predict(run_dir, selected, env))
        for split in ["test", "validation", "heldout"]:
            rows = [{"task": task.name, "controller": meta["controller"], "split": split, "rl_seed": int(meta["seed"]),
                     **run_episode(env, agent, d, s, soc)}
                    for d in split_days(env, split, proto) for s in proto["episode"]["activation_seeds"]]
            out = PARTS / task.name / meta["controller"] / f"rlseed{meta['seed']}_{split}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
            out.with_suffix(".meta.json").write_text(json.dumps(run_metadata(
                started=started, stage="rl-eval", task=task.name, controller=meta["controller"], split=split,
                seed={"rl_seed": meta["seed"], "activation_seeds": proto["episode"]["activation_seeds"]})), encoding="utf-8")
            n_final += len(rows)
    selection = {
        "controller": meta["controller"], "task": task.name, "seed": meta["seed"], "learning_rate": meta["learning_rate"],
        "tag": meta["tag"], "selection_rule": "highest mean normalised score on the validation split",
        "selected_steps": int(best.steps), "selected_checkpoint": selected.name,
        "validation_score": float(best.validation_score), "validation_cost_kw": float(best.validation_cost_kw),
        "final_episodes": n_final, "meta": run_metadata(started=started, seed=meta["seed"]),
    }
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"{run_dir.relative_to(ROOT)}: selected {int(best.steps)} steps, validation score {best.validation_score:.4f}")


if __name__ == "__main__":
    main()

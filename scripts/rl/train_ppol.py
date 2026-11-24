"""Train PPO-Lagrangian (OmniSafe ``PPOLag``, unmodified) on a benchmark task.

Run with the OmniSafe Python environment:

    python scripts/rl/train_ppol.py --task T1-Nominal --seed 0 --lr 0.0003 --cost-limit 123.4 --tag main

Reward: step profit without the penalty term. Cost: limit-violation plus
shortfall magnitude per step [kW]. ``--cost-limit`` is the cost budget per
episode (the mean episode cost of DA-OPT-MPC on the validation split).
OmniSafe defaults are used except: learning rate, ``steps_per_epoch`` (so
that checkpoints fall on the protocol's grid), one thread, CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEPS_PER_EPOCH = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--cost-limit", type=float, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--c-thr", type=float, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    import omnisafe

    from pv_bess_gym.benchmark.runinfo import run_metadata
    from pv_bess_gym.benchmark.tasks import load_protocol, results_root
    from pv_bess_gym.rl import omnisafe_env

    proto = load_protocol()
    rl = proto["rl"]
    total = int(args.total_timesteps or rl["total_timesteps"])
    run_dir = results_root() / "rl_training" / "runs" / args.tag / args.task / "da-opt-ppol" / f"lr{args.lr:g}_seed{args.seed}"
    if (run_dir / "train.json").exists():
        print(f"exists: {run_dir}")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    omnisafe_env.CONFIG.update({"split": "train", "c_thr_eur_mwh": args.c_thr, "random_soc": False,
                                "activation_seeds": rl["train_activation_seeds"]})
    save_every = max(1, int(rl["checkpoint_every"]) // STEPS_PER_EPOCH)
    custom_cfgs = {
        "seed": args.seed,
        "train_cfgs": {"total_steps": total, "torch_threads": 1, "vector_env_nums": 1, "parallel": 1, "device": "cpu"},
        "algo_cfgs": {"steps_per_epoch": STEPS_PER_EPOCH},
        "model_cfgs": {"actor": {"lr": args.lr}, "critic": {"lr": args.lr}},
        "lagrange_cfgs": {"cost_limit": float(args.cost_limit)},
        "logger_cfgs": {"use_wandb": False, "use_tensorboard": False, "save_model_freq": save_every,
                        "log_dir": str(run_dir / "omnisafe")},
    }
    started = time.time()
    agent = omnisafe.Agent("PPOLag", omnisafe_env.dispatch_env_id(args.task), custom_cfgs=custom_cfgs)
    agent.learn()
    elapsed = time.time() - started
    log_dir = sorted((run_dir / "omnisafe").glob("*/*/config.json"))[-1].parent
    config = json.loads((log_dir / "config.json").read_text(encoding="utf-8"))
    meta = {
        "library": "omnisafe", "algo": "PPOLag", "controller": "da-opt-ppol", "task": args.task, "seed": args.seed,
        "learning_rate": args.lr, "tag": args.tag, "total_timesteps": total, "steps_per_epoch": STEPS_PER_EPOCH,
        "checkpoint_every": save_every * STEPS_PER_EPOCH, "cost_limit": float(args.cost_limit),
        "train_seconds": elapsed, "steps_per_second": total / elapsed,
        "network": str(config["model_cfgs"]["actor"]["hidden_sizes"]), "discount": float(config["algo_cfgs"]["gamma"]),
        "omnisafe_log_dir": str(log_dir),
        "hyperparameters": "OmniSafe PPOLag defaults except learning rate, steps_per_epoch, cost_limit",
        "meta": run_metadata(seed=args.seed, started=started),
    }
    (run_dir / "train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{run_dir.relative_to(ROOT)}: {elapsed:.0f} s, {total / elapsed:.0f} steps/s")


if __name__ == "__main__":
    main()

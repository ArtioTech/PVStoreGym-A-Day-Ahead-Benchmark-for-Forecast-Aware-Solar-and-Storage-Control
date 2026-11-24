"""Train PPO or SAC (Stable-Baselines3, unmodified) on a benchmark task.

    python scripts/rl/train_sb3.py --task T1-Nominal --algo ppo --seed 0 --lr 0.0003 --tag main

Library defaults are used for every hyper-parameter except the learning rate.
Training episodes: one delivery day of the train split (held-out weeks
excluded), SOC 0.5, training activation seeds. A checkpoint (model and
observation/reward normalisation statistics) is saved every
``rl.checkpoint_every`` environment steps.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--algo", required=True, choices=["ppo", "sac"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--c-thr", type=float, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from pv_bess_gym.benchmark.runinfo import run_metadata
    from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, results_root, split_days
    from pv_bess_gym.rl.wrappers import DispatchRLWrapper

    torch.set_num_threads(1)
    proto = load_protocol()
    rl = proto["rl"]
    total = int(args.total_timesteps or rl["total_timesteps"])
    run_dir = results_root() / "rl_training" / "runs" / args.tag / args.task / f"da-opt-{args.algo}" / f"lr{args.lr:g}_seed{args.seed}"
    if (run_dir / "train.json").exists():
        print(f"exists: {run_dir}")
        return
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    def factory():
        env = make_task_env(args.task, c_thr_eur_mwh=args.c_thr)
        env.set_episode_days(split_days(env, "train", proto))
        wrapped = DispatchRLWrapper(env, random_soc=False, activation_seeds=rl["train_activation_seeds"])
        wrapped.reset(seed=args.seed)
        return Monitor(wrapped, filename=str(run_dir / "monitor"))

    venv = VecNormalize(DummyVecEnv([factory]), norm_obs=True, norm_reward=True)
    model = {"ppo": PPO, "sac": SAC}[args.algo]("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
                                                 learning_rate=args.lr)
    callback = CheckpointCallback(save_freq=int(rl["checkpoint_every"]), save_path=str(run_dir / "checkpoints"),
                                  name_prefix="rl_model", save_vecnormalize=True)
    started = time.time()
    model.learn(total_timesteps=total, callback=callback)
    elapsed = time.time() - started
    meta = {
        "library": "stable-baselines3", "algo": args.algo, "controller": f"da-opt-{args.algo}", "task": args.task,
        "seed": args.seed, "learning_rate": args.lr, "tag": args.tag, "total_timesteps": total,
        "checkpoint_every": int(rl["checkpoint_every"]), "train_seconds": elapsed, "steps_per_second": total / elapsed,
        "network": str(model.policy.net_arch), "discount": float(model.gamma),
        "n_parameters": int(sum(p.numel() for p in model.policy.parameters())),
        "hyperparameters": "library defaults except learning_rate; VecNormalize(norm_obs, norm_reward)",
        "meta": run_metadata(seed=args.seed, started=started),
    }
    (run_dir / "train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{run_dir.relative_to(ROOT)}: {elapsed:.0f} s, {total / elapsed:.0f} steps/s")


if __name__ == "__main__":
    main()

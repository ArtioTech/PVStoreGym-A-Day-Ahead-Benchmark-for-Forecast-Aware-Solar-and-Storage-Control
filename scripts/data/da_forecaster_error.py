"""Error of the naive DAM price forecasters per split -> results/<protocol subdir>/da_forecaster_error.json."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, results_root, split_days  # noqa: E402
from pv_bess_gym.envs.information import naive_dam_forecast  # noqa: E402

started = time.time()
env = make_task_env("T1-Nominal")
protocol = load_protocol()
out = {"meta": run_metadata(started=started), "unit": "EUR/MWh", "default": protocol["simulator"]["da_forecaster"], "methods": {}}
for method in ["prev_day", "prev_week"]:
    forecast = naive_dam_forecast(env._ts, env._dam, method)
    out["methods"][method] = {}
    for split in ["train", "validation", "test"]:
        rows = np.concatenate([np.arange(env.day_start_index(d), env.day_start_index(d) + 96) for d in split_days(env, split, protocol)])
        err = forecast[rows] - env._dam[rows]
        out["methods"][method][split] = {
            "n_steps": int(len(rows)), "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "mean_price": float(env._dam[rows].mean()),
        }
path = results_root() / "da_forecaster_error.json"   # results/<results_subdir> of the active protocol
path.write_text(json.dumps(out, indent=2), encoding="utf-8")
for method, splits in out["methods"].items():
    for split, stats in splits.items():
        print(f"{method:10s} {split:10s} MAE {stats['mae']:.2f}  RMSE {stats['rmse']:.2f}  bias {stats['bias']:+.2f}  (mean price {stats['mean_price']:.2f})")
print(f"-> {path}")

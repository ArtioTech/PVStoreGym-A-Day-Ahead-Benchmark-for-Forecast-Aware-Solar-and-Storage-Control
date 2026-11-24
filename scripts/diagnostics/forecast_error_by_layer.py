"""PV forecast error per decision layer and kappa (realism fix R5).

For each forecast series the environment can show (D-1: `_previous_day2`;
intraday/real time: short-lead up to 3 h ahead, `_previous_day1` after), and
kappa in {1, 2}: PV-power MAE, RMSE and bias against the simulated actual PV
over daylight steps (actual PV > 0), all of 2025, and GHI MAE.

    python scripts/diagnostics/forecast_error_by_layer.py
    -> results/v3/diagnostics/forecast_error_by_layer.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pv_bess_gym.benchmark.runinfo import run_metadata  # noqa: E402
from pv_bess_gym.benchmark.tasks import make_task_env, results_root  # noqa: E402

LAYER = {"d2": "D-1 (previous_day2)", "short_lead": "intraday/real time, <= 3 h ahead (short lead)",
         "d1": "intraday/real time, > 3 h ahead (previous_day1)"}


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    out = {}
    for kappa in (1.0, 2.0):
        env = make_task_env("T1-Nominal", pv_forecast_error_scale=kappa)
        actual = env._pv_act
        day = actual > 0
        ghi_act = env._ghi
        gday = ghi_act > 0
        res = {}
        for key, label in LAYER.items():
            pv, ghi, _ = env._wx[key]
            e = (pv - actual)[day]
            res[key] = {"layer": label,
                        "pv_mae_kw": float(np.abs(e).mean()), "pv_rmse_kw": float(np.sqrt((e ** 2).mean())),
                        "pv_bias_kw": float(e.mean()), "pv_mae_pct_of_mean_daylight_pv": float(np.abs(e).mean() / actual[day].mean() * 100),
                        "ghi_mae_wm2": float(np.abs(ghi - ghi_act)[gday].mean())}
        res["_mean_daylight_pv_kw"] = float(actual[day].mean())
        out[f"kappa_{kappa:g}"] = res
    out["meta"] = run_metadata(started=started)
    path = results_root() / "diagnostics" / "forecast_error_by_layer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k != "meta":
            print(k, {kk: (round(vv["pv_mae_kw"], 2), round(vv["pv_mae_pct_of_mean_daylight_pv"], 1), round(vv["pv_bias_kw"], 2)) for kk, vv in v.items() if isinstance(vv, dict)})
    print("->", path)


if __name__ == "__main__":
    main()

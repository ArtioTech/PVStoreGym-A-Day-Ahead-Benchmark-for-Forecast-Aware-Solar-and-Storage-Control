"""rev1 vs v3 on T1-Nominal, test split: ORACLE and the five baselines (no ORACLE-DELIVER, no RL).

    .venv/bin/python results/rev1/impact_T1.py   -> results/rev1/impact_T1.json
IQM = 25% trimmed mean over episodes (as rliable.metrics.aggregate_iqm); CIs: day bootstrap
(5,000 resamples, seed 0) for profits, episode bootstrap (5,000, seed 0) for IQM.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import trim_mean

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["day", "activation_seed"]
ORDER = ["da-only", "da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc"]


def day_boot(df, col):
    per = df.groupby("day")[col].mean().to_numpy()
    r = np.random.default_rng(0)
    d = per[r.integers(len(per), size=(5000, len(per)))].mean(axis=1)
    return [float(per.mean()), *map(float, np.percentile(d, [2.5, 97.5]))]


def iqm_boot(x):
    x = np.asarray(x)
    r = np.random.RandomState(0)
    d = [trim_mean(x[r.randint(len(x), size=len(x))], 0.25) for _ in range(5000)]
    return [float(trim_mean(x, 0.25)), *map(float, np.percentile(d, [2.5, 97.5]))]


def rev1_frame():
    parts = ROOT / "results" / "rev1" / "parts"
    orc = pd.concat([pd.read_csv(p) for p in sorted((parts / "oracle").rglob("test_*.csv"))])
    runs = pd.concat([pd.read_csv(p) for p in sorted((parts / "T1-Nominal").rglob("test_*.csv"))])
    ref = runs[runs.controller == "da-only"][KEYS + ["profit_eur"]].rename(columns={"profit_eur": "J_L"})
    df = runs.merge(orc[KEYS + ["U", "U_rep", "mip_gap"]], on=KEYS).merge(ref, on=KEYS)
    assert not (df.profit_eur > df.U + 1e-6).any(), "profit above U"
    df["score_valid"] = (df.U - df.J_L) >= 1.0
    df["score"] = np.where(df.score_valid, (df.profit_eur - df.J_L) / (df.U - df.J_L), np.nan)
    return df


def main():
    v3 = pd.read_csv(ROOT / "results" / "v3" / "benchmark_T1-Nominal.csv")
    v3 = v3[(v3.split == "test") & v3.controller.isin(ORDER)]
    out = {}
    for name, df in [("v3", v3), ("rev1", rev1_frame())]:
        ref = df[df.controller == "da-only"]
        e = {"episodes_per_controller": int(len(ref)), "U": day_boot(ref, "U"), "U_rep": day_boot(ref, "U_rep"),
             "J_L": day_boot(ref, "J_L"), "max_mip_gap": float(ref.mip_gap.max()),
             "episodes_excluded": int((~ref.score_valid).sum()), "controllers": {}}
        for c in ORDER:
            r = df[df.controller == c]
            e["controllers"][c] = {
                "S_iqm": iqm_boot(r[r.score_valid].score), "S_mean": float(r[r.score_valid].score.mean()),
                "profit": day_boot(r, "profit_eur"),
                "components": {k: float(r[k].mean()) for k in ["dam_eur", "id_eur", "afrr_capacity_eur",
                               "afrr_activation_eur", "imbalance_eur", "degradation_eur", "penalties_eur"]},
                "scheduled_kwh": float(r.scheduled_kwh.mean()), "metered_kwh": float(r.metered_kwh.mean()),
                "share_short": float(r.short_imbalance_steps.sum() / (96 * len(r))),
                "share_long": float(r.long_imbalance_steps.sum() / (96 * len(r))),
            }
        out[name] = e
    (ROOT / "results" / "rev1" / "impact_T1.json").write_text(json.dumps(out, indent=2))
    for name, e in out.items():
        print(f"\n{name}: U {e['U'][0]:.1f}  U_rep {e['U_rep'][0]:.1f}  J_L {e['J_L'][0]:.1f}  max gap {e['max_mip_gap']:.5f}  excluded {e['episodes_excluded']}")
        for c, v in e["controllers"].items():
            s, p, k = v["S_iqm"], v["profit"], v["components"]
            print(f"  {c:14s} S IQM {s[0]:+.4f} [{s[1]:+.4f},{s[2]:+.4f}]  profit {p[0]:7.1f} [{p[1]:.1f},{p[2]:.1f}]  "
                  f"DAM {k['dam_eur']:6.1f} ID {k['id_eur']:7.1f} imb {k['imbalance_eur']:7.1f} cap {k['afrr_capacity_eur']:5.1f} act {k['afrr_activation_eur']:5.1f} pen {k['penalties_eur']:4.1f}  "
                  f"sched {v['scheduled_kwh']:6.1f} meter {v['metered_kwh']:6.1f}")


if __name__ == "__main__":
    main()

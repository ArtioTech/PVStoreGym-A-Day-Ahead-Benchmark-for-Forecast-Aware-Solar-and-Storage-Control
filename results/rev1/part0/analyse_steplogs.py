"""Part 0 (rev1 brief): tables from the per-step logs of steplog.py (T1-Nominal, test split).

    .venv/bin/python results/rev1/part0/analyse_steplogs.py
Writes results/rev1/part0/part0_tables.json and prints the tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P_EXP = P_IMP = 373.0
DT = 0.25
ORDER = ["da-only", "da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc", "da-opt-ppo", "da-opt-sac", "da-opt-ppol"]


def load(controller: str) -> pd.DataFrame:
    files = sorted((HERE / os.environ.get("STEPLOG_DIR", "steplogs")).glob(f"{controller}_rl*_*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["hour"] = pd.to_datetime(df.timestamp.str.slice(0, 19)).dt.hour     # local Europe/Athens wall clock
    df["q_kw"] = df.x_kw + df.y_kw
    return df


def main() -> None:
    out: dict = {}
    for c in ORDER:
        df = load(c)
        if df.empty:
            continue
        n_ep = df.groupby(["rl_seed", "day", "activation_seed"]).ngroups
        e = {"episodes": int(n_ep), "steps": int(len(df))}
        # 0.1 hourly profile
        hourly = df.groupby("hour").agg(y_kw=("y_kw", "mean"), x_kw=("x_kw", "mean"), pv_fcst_kw=("pv_fcst_kw", "mean"),
                                        d1_plan_battery_kw=("d1_plan_battery_kw", "mean"), q_kw=("q_kw", "mean"),
                                        metered_kw=("metered_kw", "mean"))
        e["hourly"] = {int(h): {k: float(v) for k, v in r.items()} for h, r in hourly.iterrows()}
        ch = df[df.d1_plan_battery_kw < -1.0]
        di = df[df.d1_plan_battery_kw > 1.0]
        idle = df[df.d1_plan_battery_kw.abs() <= 1.0]
        e["y_by_d1_plan_state"] = {
            "plan_charging": {"share_steps": float(len(ch) / len(df)), "mean_y_kw": float(ch.y_kw.mean()),
                              "mean_plan_battery_kw": float(ch.d1_plan_battery_kw.mean())},
            "plan_discharging": {"share_steps": float(len(di) / len(df)), "mean_y_kw": float(di.y_kw.mean()),
                                 "mean_plan_battery_kw": float(di.d1_plan_battery_kw.mean())},
            "plan_idle": {"share_steps": float(len(idle) / len(df)), "mean_y_kw": float(idle.y_kw.mean())},
        }
        e["mean_y_kw"] = float(df.y_kw.mean())
        e["id_energy_net_kwh_per_day"] = float(df.y_kw.sum() * DT / n_ep)
        e["id_energy_abs_kwh_per_day"] = float(df.y_kw.abs().sum() * DT / n_ep)
        # 0.2 intraday price information (steps with an intraday position)
        traded = df[(df.decided_at >= 0) & (df.y_kw.abs() > 1e-9)].copy()
        if len(traded):
            traded["known"] = (traded.seen_value - traded.price_id_settle).abs() < 1e-6
            w = traded.y_kw.abs()
            e["intraday_price_information"] = {
                "share_abs_energy_price_known_at_decision": float((w * traded.known).sum() / w.sum()),
                "share_steps_price_known_at_decision": float(traded.known.mean()),
                "abs_energy_kwh_per_day": float(w.sum() * DT / n_ep),
                "by_settlement_auction": {
                    a: {"share_abs_energy": float(w[traded.settle_auction == a].sum() / w.sum()),
                        "share_known": float((w * traded.known)[traded.settle_auction == a].sum() / max(w[traded.settle_auction == a].sum(), 1e-12))}
                    for a in sorted(traded.settle_auction.unique())},
                "unknown_by_hour_share_abs_energy": {int(h): float(v) for h, v in
                                                     (w[~traded.known].groupby(traded.hour[~traded.known]).sum() / w.sum()).items()},
                "seen_label_when_unknown": {str(k): int(v) for k, v in traded.seen_label[~traded.known].value_counts().items()},
                "mean_abs_price_error_when_unknown_eur_mwh": float((traded.seen_value - traded.price_id_settle).abs()[~traded.known].mean())
                if (~traded.known).any() else 0.0,
            }
        # 0.5 commercial position limits
        over = (df.q_kw - P_EXP).clip(lower=0.0)
        under = (-P_IMP - df.q_kw).clip(lower=0.0)
        e["position_limits"] = {
            "share_steps_q_above_P_exp": float((df.q_kw > P_EXP + 1e-6).mean()),
            "share_steps_q_below_minus_P_imp": float((df.q_kw < -P_IMP - 1e-6).mean()),
            "energy_above_P_exp_kwh_per_day": float(over.sum() * DT / n_ep),
            "energy_below_minus_P_imp_kwh_per_day": float(under.sum() * DT / n_ep),
            "max_q_kw": float(df.q_kw.max()), "min_q_kw": float(df.q_kw.min()),
            "share_steps_x_below_0": float((df.x_kw < -1e-6).mean()),
        }
        # 0.6 committed reserve
        daily = df.groupby(["rl_seed", "day", "activation_seed"])[["r_up_kw", "r_dn_kw"]].first()
        e["committed_reserve_kw"] = {"r_up_mean": float(daily.r_up_kw.mean()), "r_dn_mean": float(daily.r_dn_kw.mean()),
                                     "share_days_r_up_positive": float((daily.r_up_kw > 1e-6).mean()),
                                     "share_days_r_dn_positive": float((daily.r_dn_kw > 1e-6).mean())}
        out[c] = e
    (HERE / os.environ.get("STEPLOG_TABLES", "part0_tables.json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c, e in out.items():
        print(f"\n== {c}: {e['episodes']} episodes; mean y {e['mean_y_kw']:+.2f} kW; ID net {e['id_energy_net_kwh_per_day']:+.1f} kWh/day")
        print("   by D-1 plan state:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in e["y_by_d1_plan_state"].items()})
        if "intraday_price_information" in e:
            ipi = e["intraday_price_information"]
            print(f"   price known share (|y| weighted) {ipi['share_abs_energy_price_known_at_decision']:.4f}; by auction {ipi['by_settlement_auction']}")
            print(f"   unknown by hour {ipi['unknown_by_hour_share_abs_energy']}; labels {ipi['seen_label_when_unknown']}")
        print("   limits:", {k: round(v, 4) for k, v in e["position_limits"].items()})
        print("   reserve:", {k: round(v, 3) for k, v in e["committed_reserve_kw"].items()})


if __name__ == "__main__":
    main()

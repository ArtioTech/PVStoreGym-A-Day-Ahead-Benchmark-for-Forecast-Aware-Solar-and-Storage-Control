"""Markdown tables of paper_inputs_rev1.md from results/rev1 (summary JSONs, oracle CSVs).

    .venv/bin/python results/rev1/make_tables.py > results/rev1/tables_rev1.md

Every number is read from a result file; nothing is typed in. The v3 comparison
(Table A2) is written only while results/v3/ exists.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
REV1, V3 = R / "rev1", R / "v3"
TASKS = ["T1-Nominal", "T2-Forecast", "T3-Gate", "T4-Ageing", "T5-Combined"]
CTRL = ["da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc", "da-opt-ppo", "da-opt-sac", "da-opt-ppol"]
NAME = {"da-only": "DA-ONLY", **{c: c.upper() for c in CTRL}}
AVAIL = [t for t in TASKS if (REV1 / f"summary_{t}.json").exists()]
S = {t: json.loads((REV1 / f"summary_{t}.json").read_text()) for t in AVAIL}
S3 = {t: json.loads((V3 / f"summary_{t}.json").read_text()) for t in TASKS if (V3 / f"summary_{t}.json").exists()}
KEYS = ["dam_eur", "id_eur", "afrr_capacity_eur", "afrr_activation_eur", "imbalance_eur", "degradation_eur", "penalties_eur"]


def ci(e, d=4, sign=True):
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f"{f.format(e['value'])} [{f.format(e['ci95'][0])}, {f.format(e['ci95'][1])}]".replace("-", "−")


def eur(e):
    return ci(e, 1, False)


def row(label, cells):
    print("| " + label + " | " + " | ".join(cells) + " |")


def head(cols):
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))


def day_boot(per_day, reps=5000, seed=0):
    per_day = np.asarray(per_day, dtype=float)
    rng = np.random.default_rng(seed)
    d = per_day[rng.integers(len(per_day), size=(reps, len(per_day)))].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"value": float(per_day.mean()), "ci95": [float(lo), float(hi)]}


def table_a(split="test"):
    print(f"## Table A — anchors and scores, {split} split\n")
    head([""] + AVAIL)
    A = {t: S[t]["splits"][split]["anchors"] for t in AVAIL}
    row("ORACLE bound U", [eur(A[t]["U_mean_daily_eur"]) for t in AVAIL])
    row("Replayed ORACLE U_rep", [eur(A[t]["U_rep_mean_daily_eur"]) for t in AVAIL])
    row("(U − U_rep)/U", [f"{A[t]['relative_gap_U_minus_U_rep_over_U']:.4f}" for t in AVAIL])
    row("Max MIP gap, ORACLE", [f"{A[t]['max_mip_gap']:.4f}" for t in AVAIL])
    row("ORACLE-DELIVER bound U_D", [eur(A[t]["U_D_mean_daily_eur"]) if "U_D_mean_daily_eur" in A[t] else "not run" for t in AVAIL])
    row("DA-ONLY J_L", [eur(A[t]["J_L_mean_daily_eur"]) for t in AVAIL])
    row("Episodes with U − J_L < 1 EUR", [str(A[t]["episodes_excluded_from_score"]) for t in AVAIL])
    row("Episodes with U_D − J_L < 1 EUR", [str(A[t].get("episodes_excluded_from_score_D", "–")) for t in AVAIL])
    row("Episodes with U_D < J_L", [str(A[t].get("episodes_U_D_below_J_L", "–")) for t in AVAIL])
    row("Max MIP gap, ORACLE-DELIVER", [f"{A[t]['max_mip_gap_D']:.4f}" if "max_mip_gap_D" in A[t] else "–" for t in AVAIL])
    for key, lab in (("score", "S"), ("score_D", "S_D")):
        print(f"| **IQM of {lab}** |" + " |" * len(AVAIL))
        for c in CTRL:
            cells = []
            for t in AVAIL:
                e = S[t]["splits"][split]["controllers"].get(c, {})
                cells.append(ci(e[key]["iqm"]) if key in e else "–")
            row(NAME[c], cells)
    print()


def table_a2():
    if not S3:
        return
    print("## Table A2 — IQM of S and J_L, v3 → rev1 (test)\n")
    head(["Baseline"] + [f"{t} v3 → rev1" for t in AVAIL])
    for c in CTRL:
        cells = []
        for t in AVAIL:
            a = S3[t]["splits"]["test"]["controllers"][c]["score"]["iqm"]["value"]
            b = S[t]["splits"]["test"]["controllers"][c]["score"]["iqm"]["value"]
            cells.append(f"{a:+.4f} → {b:+.4f}".replace("-", "−"))
        row(NAME[c], cells)
    for lab, key in (("U [EUR/day]", "U_mean_daily_eur"), ("J_L [EUR/day]", "J_L_mean_daily_eur")):
        row(lab, [f"{S3[t]['splits']['test']['anchors'][key]['value']:.1f} → {S[t]['splits']['test']['anchors'][key]['value']:.1f}" for t in AVAIL])
    print()


def table_b(t="T1-Nominal"):
    print(f"## Table B — secondary metrics, {t} test (per day)\n")
    head(["Baseline", "Mean profit [EUR]", "CVaR₅% [EUR]", "Reserve deliverability", "Penalties [EUR]", "Cost [kW]",
          "Equiv. full cycles", "Curtailment [kWh]", "Decision time [ms/step]", "Mean sign of imbalance", "corr(imbalance, ID − IMB price)"])
    for c in ["da-only"] + CTRL:
        e = S[t]["splits"]["test"]["controllers"][c]
        im = e["imbalance"]
        row(NAME[c], [f"{e['mean_daily_profit_eur']['value']:.1f}", f"{e['cvar5_daily_profit_eur']:.1f}", f"{e['reserve_deliverability']:.4f}",
                      f"{e['penalties_eur_per_day']:.2f}", f"{e['cost_kw_per_day']:.1f}", f"{e['equivalent_full_cycles_per_day']:.3f}",
                      f"{e['curtailment_kwh_per_day']:.1f}", f"{e['decision_time_ms_per_step']:.2f}",
                      f"{im['mean_sign_of_step_imbalance']:+.3f}", f"{im['correlation_with_id_minus_imbalance_price']:+.3f}"])
    print()


def table_b2(t="T1-Nominal"):
    print(f"## Table B2 — profit by component, {t} test, EUR/day (degradation and penalties are costs, subtracted)\n")
    head(["Controller", "DAM", "Intraday", "Capacity", "Activation", "Imbalance", "Degradation", "Penalties"])
    for c in ["da-only"] + CTRL:
        comp = S[t]["splits"]["test"]["controllers"][c]["components_eur_per_day"]
        row(NAME[c], [f"{comp[k]:.1f}" for k in KEYS])
    o = pd.read_csv(REV1 / f"oracle_{t}.csv")
    o = o[o.split == "test"]
    row("ORACLE (incumbent)", [f"{o[f'oracle_{k}'].mean():.1f}" for k in KEYS])
    if "oracle_D_dam_eur" in o.columns and o.oracle_D_dam_eur.notna().all():
        row("ORACLE-DELIVER (incumbent)", [f"{o[f'oracle_D_{k}'].mean():.1f}" for k in KEYS])
    print()


def table_b3(t="T1-Nominal"):
    print(f"## Table B3 — scheduled vs metered energy, {t} test\n")
    head(["Controller", "Scheduled [kWh/day]", "Metered [kWh/day]", "Mean imbalance [kWh/day]", "Imbalance [EUR/day]", "Steps short", "Steps long"])
    for c in ["da-only"] + CTRL:
        e = S[t]["splits"]["test"]["controllers"][c]
        d = e["delivery"]
        row(NAME[c], [f"{d['scheduled_kwh_per_day']:.1f}", f"{d['metered_kwh_per_day']:.1f}", f"{e['imbalance']['mean_imbalance_kwh_per_day']:.1f}",
                      f"{e['components_eur_per_day']['imbalance_eur']:.1f}", f"{100 * d['share_steps_short']:.1f}%", f"{100 * d['share_steps_long']:.1f}%"])
    print()


def table_c():
    print("## Table C — IQM of S by split (validation / held-out weeks / test)\n")
    head(["Baseline"] + AVAIL)
    for c in CTRL:
        cells = []
        for t in AVAIL:
            sp = S[t]["splits"]
            cells.append(" / ".join(f"{sp[s]['controllers'][c]['score']['iqm']['value']:+.4f}".replace("-", "−")
                                    for s in ("validation", "heldout", "test")))
        row(NAME[c], cells)
    print("\nValidation was used for RL checkpoint selection (optimistic for RL); held-out weeks were not.\n")


def table_e():
    print("## Table E — 2×2 layer attribution (paired episodes, day bootstrap)\n")
    head(["Task", "Score", "paired", "Δ_ID", "Δ_RT", "interaction"])
    for t in AVAIL:
        for key, lab in (("layer_attribution", "S"), ("layer_attribution_D", "S_D")):
            la = S[t]["splits"]["test"].get(key)
            if la:
                row(t, [lab, str(la["n_paired_episodes"]), ci(la["delta_id"]), ci(la["delta_rt"]), ci(la["interaction"])])
    print()


def table_h():
    print("## Table H — ORACLE-DELIVER bound and incumbent, test split\n")
    head(["Task", "U_D (bound)", "U_D incumbent", "median gap", "p90 gap", "max gap", "gap > 5%", "gap > 1%", "solves ≥ 120 s", "mean solve [s]", "retries"])
    for t in AVAIL:
        o = pd.read_csv(REV1 / f"oracle_{t}.csv")
        o = o[(o.split == "test")]
        if "U_D" not in o.columns or o.U_D.isna().all():
            row(t, ["not run"] + ["–"] * 9)
            continue
        g = o.mip_gap_D
        row(t, [eur(day_boot(o.groupby("day").U_D.mean())), eur(day_boot(o.groupby("day").U_D_incumbent.mean())),
                f"{g.median():.5f}", f"{g.quantile(0.9):.5f}", f"{g.max():.5f}", str(int((g > 0.05).sum())), str(int((g > 0.01).sum())),
                f"{100 * (o.solve_seconds_D >= 120).mean():.1f}%", f"{o.solve_seconds_D.mean():.1f}",
                str(int(o.get("retries_D", pd.Series([0])).fillna(0).sum()))])
    print()


def table_i():
    print("## Table I — mean daily profit, test split, EUR/day [95% day-bootstrap CI]\n")
    head(["Controller"] + AVAIL)
    for c in ["da-only"] + CTRL:
        row(NAME[c], [eur(S[t]["splits"]["test"]["controllers"][c]["mean_daily_profit_eur"]) for t in AVAIL])
    row("ORACLE bound U", [eur(S[t]["splits"]["test"]["anchors"]["U_mean_daily_eur"]) for t in AVAIL])
    row("ORACLE-DELIVER bound U_D", [eur(S[t]["splits"]["test"]["anchors"]["U_D_mean_daily_eur"])
                                     if "U_D_mean_daily_eur" in S[t]["splits"]["test"]["anchors"] else "not run" for t in AVAIL])
    print()


def table_rl():
    print("## RL details (test)\n")
    head(["Task", "Std of per-seed IQM S (PPO / SAC / PPOL)", "PPOL cost [kW/day]", "Reserve deliverability (PPO / SAC / PPOL)",
          "Penalties [EUR/day] (PPO / SAC / PPOL)", "Mean S (PPO / SAC / PPOL)"])
    for t in AVAIL:
        cs = S[t]["splits"]["test"]["controllers"]
        rl = CTRL[4:]
        row(t, [" / ".join(f"{cs[c]['score_iqm_std_over_rl_seeds']:.4f}" for c in rl), f"{cs['da-opt-ppol']['cost_kw_per_day']:.1f}",
                " / ".join(f"{cs[c]['reserve_deliverability']:.4f}" for c in rl), " / ".join(f"{cs[c]['penalties_eur_per_day']:.2f}" for c in rl),
                " / ".join(f"{cs[c]['score']['mean']['value']:+.4f}".replace("-", "−") for c in rl)])
    print()


if __name__ == "__main__":
    print(f"<!-- generated by results/rev1/make_tables.py from results/rev1/ ({', '.join(AVAIL)}) -->\n")
    for fn in (table_a, table_a2, table_i, table_b, table_b2, table_b3, table_c, table_e, table_h, table_rl):
        fn()

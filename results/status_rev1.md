# Status: rev1 brief, Part 0 (verification, nothing changed)

Brief: "final technical revision" (rev1). Code at commit `e854535` (protocol v3,
`configs/journal.yaml`, results in `results/v3/`). Part 0 is read-only: no
library code, config or result file was changed. The only new files are under
`results/rev1/part0/` (diagnostic scripts, logs, step logs) and this file.

Verdicts: **done** = the code already does what the manuscript/brief requires;
**issue** = it does not (input for Part A); **n/a** = not applicable.

| Item | Verdict | One line |
|---|---|---|
| 0.1 Intraday reference | **issue** | Both intraday baselines use the PV forecast only; the D-1 plan's battery power is ignored, so they sell when the plan charges |
| 0.2 Intraday price information | **issue (small)** | 89.8–96.4% of traded intraday energy is priced at a value known at decision time; the rest is the 12:00–13:00 block, settled at IDA3 but decided before IDA3 is published |
| 0.3 IDA3 coverage / time zones | **issue (major)** | All HEnEx and IPTO market files are in CET/CEST; the code reads them as Athens time, so **every market price is one hour early** relative to PV. IDA3 appears at local 12:00–24:00 instead of 13:00–01:00 |
| 0.4 DST days | done | 2025-03-30 (92 steps) and 2025-10-26 (100 steps) are skipped |
| 0.5 Position limits | **issue** | `q = x + Y` is not bounded by ±373 kW; up to 24% of steps exceed `P_exp` |
| 0.6 Oracle vs D-1 constraints | done (list) | Eight D-1 constraints/terms are absent from the oracles; see list and r± |
| 0.7 Upper-bound argument | done | `η̄ = 0.97` = maximum of the environment efficiency; self-discharge order is the same in both; no episode above `U` |
| 0.8 ORACLE-DELIVER incumbents | done (reported) | T1: bound 575.11, incumbent 574.26 EUR/day; 3 episodes with gap > 5% |
| 0.9 Table 14 timings | **issue (manuscript)** | Both numbers are computed correctly; mean of per-run rates vs mean time diverge because of slow T5 runs |
| 0.10 Observations | done (listed) | RL sees x, Y, r±, published ID price, 8-step forecast; not the full-day context or the D-1 battery plan |
| 0.11 Statistics | done (reported) | IQM bootstrap resamples episodes (not days); profits and attribution resample days; seeds shared |
| extra: reproducibility | **issue (small)** | RL test profits depend on the order of D-1 QP solves in a process (warm start), up to 8.98 EUR per episode, ≤ 0.014 EUR/day in means |
| extra: ORACLE-DELIVER reproducibility | **issue (small)** | Re-solving 91 T1 episodes gives bounds up to 9.53 EUR away from `results/v3` (120 s time limit, load-dependent) |
| extra: weather time stamps | **issue (to check)** | Irradiance (actual and forecasts) peaks 22–25 min after solar noon; consistent with hourly values labelled at the end of their averaging hour |

Compute used in Part 0: step logs 1.33 CPU-h (baselines, 4,796 s user) +
0.18 CPU-h (RL) + 1.95 CPU-h ORACLE re-solve + 1.28 CPU-h ORACLE-DELIVER
re-solve (user time) ≈ 4.7 CPU-h in total. Far below the 120 CPU-h ceiling.

Commands (all exit status 0):

```
.venv/bin/python results/rev1/part0/steplog.py baselines --jobs 13          # 7.0 min wall
.venv/bin/python results/rev1/part0/steplog.py rl --controllers da-opt-ppo da-opt-sac --jobs 3   # 2.6 min
.venv-omnisafe/bin/python results/rev1/part0/steplog.py rl --controllers da-opt-ppol --jobs 3    # 1.2 min
.venv/bin/python results/rev1/part0/analyse_steplogs.py                       # -> part0_tables.json
.venv/bin/python results/rev1/part0/oracle_reserve.py oracle --seeds 0 1 2 3 4 5 6 7 8 9 --jobs 11
.venv/bin/python results/rev1/part0/oracle_reserve.py deliver --seeds 0 --jobs 13
```

The step-log run reproduces `results/v3/benchmark_T1-Nominal.csv`: baselines
4,550 episodes, max |Δprofit| = 5.7e-14 EUR. RL: see "extra: reproducibility".

---

## 0.1 Intraday baselines vs the D-1 plan — issue

**What the code uses as expected delivery.**

- Rule (DA-OPT-RBC, DA-OPT-RT-MPC), `_rbc_intraday_action`,
  [day_ahead_baselines.py:441-456](../pv_bess_gym/agents/day_ahead_baselines.py):
  `y = 0.98 · min(mean p̂PV over T(k), P_exp) − mean x over T(k)`.
  PV forecast only. `Y` of earlier blocks does not enter because every
  delivery step belongs to exactly one intraday block; the environment
  *replaces* (does not add to) the correction for that block
  ([pv_bess_greece_env.py:1277](../pv_bess_gym/envs/pv_bess_greece_env.py)).
- MPC (DA-OPT-ID-MPC, DA-OPT-MPC), `_solve_intraday_mpc`,
  [day_ahead_baselines.py:659-740](../pv_bess_gym/agents/day_ahead_baselines.py):
  maximise `Σ λ̂ID_k' y dt − Σ ρ_k' |p̂PV_k' − x_k' − y| dt` over the 4 steps of the
  target block (horizon = min(block length, 8) = 4), `ρ_k' = max(140, |λ̂imb_k'|)`,
  where under the information filter `λ̂imb` is the observed DAM price
  ([pv_bess_greece_env.py:478](../pv_bess_gym/envs/pv_bess_greece_env.py)).
  PV forecast only; the docstring states the assumption "battery tracks the
  commercial schedule".
- `p̂PV` is the newest forecast at decision time (column 0 of
  `day_ahead_context`, overwritten with the `id_weather=latest` series,
  [pv_bess_greece_env.py:1324-1326](../pv_bess_gym/envs/pv_bess_greece_env.py)).
- Since R4 the D-1 schedule `x` equals the planned *metered* injection
  including planned battery power ([day_ahead_baselines.py:331](../pv_bess_gym/agents/day_ahead_baselines.py)).
  Neither intraday layer sees the planned battery power, so the inconsistency
  the brief describes is present.

**T1 test, 910 episodes, mean by local hour** (`part0_tables.json`, key `hourly`):

| Hour | DA-OPT-MPC `y` | DA-OPT-RBC `y` | `x` | `p̂PV` (at step) | D-1 planned battery (+ = discharge) |
|---:|---:|---:|---:|---:|---:|
| 6 | 53.5 | −61.4 | 61.4 | 0.0 | 62.0 |
| 7 | 88.7 | −89.9 | 91.5 | 1.6 | 90.8 |
| 8 | 66.0 | −30.7 | 53.3 | 23.1 | 30.6 |
| 9 | 79.0 | 20.3 | 57.0 | 78.9 | −22.2 |
| 10 | 123.3 | 87.3 | 56.9 | 147.2 | −91.3 |
| 11 | 175.4 | 145.3 | 52.3 | 201.6 | −149.6 |
| 12 | 180.1 | 143.8 | 79.4 | 227.8 | −148.0 |
| 13 | 156.8 | 92.6 | 131.0 | 228.1 | −94.5 |
| 14 | 120.9 | 32.8 | 171.7 | 208.7 | −31.2 |
| 15 | 142.6 | −5.9 | 174.1 | 171.6 | 8.6 |
| 16 | 139.3 | −86.8 | 212.0 | 127.7 | 90.4 |
| 17 | 134.3 | −113.7 | 157.1 | 44.3 | 114.7 |
| 18 | 85.3 | −123.1 | 139.1 | 16.3 | 123.9 |
| 19 | 85.2 | −113.7 | 115.1 | 1.4 | 114.8 |
| 20 | 75.9 | −58.9 | 58.9 | 0.0 | 59.5 |
| 21 | 24.6 | −15.1 | 15.1 | 0.0 | 15.2 |

Hours 0–5 and 22–23: |y| ≤ 16.4 kW for MPC; RBC −46.2 kW at 23:00 (full table in
`part0_tables.json`). kW, mean over all steps of the hour.

By state of the D-1 plan (all steps):

| Controller | plan charging (30.4% of steps, mean −88.6 kW) | plan discharging (21.8%, +172.0 kW) | plan idle (47.7%) |
|---|---:|---:|---:|
| DA-OPT-RBC / -RT-MPC | +64.9 | −103.4 | −15.3 |
| DA-OPT-ID-MPC / -MPC | +136.8 | +89.7 | +24.3 |
| PPO (5 seeds) | +2.7 | +47.6 | +101.9 |
| SAC (5 seeds) | +115.0 | +180.0 | +137.9 |
| PPO-Lagrangian (5 seeds) | +68.2 | +76.7 | +112.2 |

Mean `y` in kW. Reading: in hours where the plan charges the battery, the rule
sells about the planned charging power (the PV forecast exceeds `x` by it), and
in the evening it buys back the planned discharge; it therefore works against
the D-1 plan in both directions. The MPC sells in charging hours too, and also
sells in discharging/evening hours (see 0.5: whenever `λ̂ID > max(140, |λ̂DAM|)`
its model makes selling the full intraday range profitable).

## 0.2 Intraday price information — issue (small)

- Settlement price: `price_id_eur_mwh` of the monthly file = the latest auction
  covering the period, IDA3 > IDA2 > IDA1, latest file version
  ([greek_2025.py:956-960](../pv_bess_gym/data/builders/greek_2025.py)), applied in
  [pv_bess_greece_env.py:677](../pv_bess_gym/envs/pv_bess_greece_env.py).
- `λ̂ID` used by the intraday MPC: column 2 of `day_ahead_context`, which under the
  filter is the newest auction *published* at the decision step
  ([pv_bess_greece_env.py:455-481, 1318](../pv_bess_gym/envs/pv_bess_greece_env.py)).
- Publication steps on the test days (`publication_times_2025.csv`): IDA1 and IDA2
  on D-1 (all steps ≤ −1), IDA3 at 11:22–11:27 local = step 46. IDA1 missing on
  4 of 92 test days; IDA2 present on all.
- Measured on T1 test (all episodes; "known" = the value observed at the
  decision step equals the settlement price to 1e-6):

| Controller | Share of \|y\|-energy with price known | Of IDA3-settled energy | Unknown energy, by hour |
|---|---:|---:|---|
| DA-OPT-RBC / -RT-MPC | 0.8978 | 0.8442 | all at 12:00–13:00 |
| DA-OPT-ID-MPC / -MPC | 0.9138 | 0.8755 | all at 12:00–13:00 |
| PPO | 0.9627 | 0.9245 | all at 12:00–13:00 |
| SAC | 0.9643 | 0.9372 | all at 12:00–13:00 |
| PPO-Lagrangian | 0.9459 | 0.8987 | all at 12:00–13:00 |

For those trades the observed value is the IDA2 price (label `ida2` in every
case). For the known trades `λ̂ID` equals the settlement price by construction.
The unknown block arises because the 12:00–13:00 block is decided at 11:00
(gate 60 min), before IDA3 is published, while settlement uses IDA3. **With the
0.3 correction the dataset's IDA3 coverage moves to 13:00–01:00; the block
13:00–14:00 is decided at 12:00, after publication, so this item may disappear
once 0.3 is fixed.** Not yet measured.

## 0.3 IDA3 coverage and time zones — issue (major)

**How the code maps market files.** HEnEx DAM, IDA1–3:
`timestamp = 00:00 Europe/Athens of the file date + (SORT − 1) · DELIVERY_DURATION`
([greek_2025.py:1127-1132](../pv_bess_gym/data/builders/greek_2025.py); same rule
in the sidecar builder [publication.py `_henex_day`](../pv_bess_gym/data/builders/publication.py)).
IPTO IMBABE (imbalance price): `STARTDATE` localised as Europe/Athens
([greek_2025.py:988-990](../pv_bess_gym/data/builders/greek_2025.py)). IPTO ISP1 aFRR
capacity prices and BalancingEnergy/CapacityProduct: localised as Europe/Athens
([ipto_balancing.py `_localize`](../pv_bess_gym/data/builders/ipto_balancing.py)).
IDA3 coverage comes from the price files (NaN where an auction has no MTU),
not from a clock rule. No CET → EET conversion is done anywhere.

**Evidence that the files are in CET/CEST** (raw files under `data/raw/`):

| File | 2025-03-30 (spring) | 2025-10-26 (autumn) |
|---|---|---|
| HEnEx DAM, IDA1 `DELIVERY_MTU` | 23 MTUs; 01:00 is followed by **03:00** (02:00 skipped) | 100 MTUs; **02:00–02:45 appears twice** (SORT 9–12 and 13–16) |
| IPTO IMBABE `STARTDATE` | — | 02:00–02:45 appears twice |
| IPTO ISP1 aFRR header | — | 02:00–02:45 appears twice |

CET/CEST skips 02:00 and repeats 02:00; Athens (EET/EEST) skips and repeats
03:00 (both change at 01:00 UTC). The files are therefore in CET/CEST wall
clock. IDA3 on a normal day starts at SORT 49 (15 min) / 13 (60 min), labelled
`12:00`, i.e. 12:00 CET, matching the SIDC IDA3 coverage of 12:00–24:00 CET.

**Consequence.** Every DAM, intraday, imbalance and aFRR capacity price is
placed one hour earlier than its delivery period in Athens time: the value at
local `h` belongs to local `h + 1`; the first local hour of each day carries
the price of 01:00–02:00, and the price of 00:00–01:00 local (23:00–24:00 CET
of D−1) is never used. PV and weather are correct in Athens time (Open-Meteo is
requested in GMT and converted, [greek_2025.py:307,317](../pv_bess_gym/data/builders/greek_2025.py)).
IDA3 coverage in the dataset (test split, share of steps with an IDA3 price by
local hour): 0.0 for hours 0–11, 1.0 for hours 12–23. **Local 12:00–13:00 is
wrongly assigned an IDA3 price**, and so is every other hour, shifted by one.

Size on the test split (91 days): PV-actual × DAM price, as aligned now
129.48 EUR/day, with prices moved one hour later 115.71 EUR/day (−13.77);
mean |DAM(k) − DAM(k − 1 h)| = 19.50 EUR/MWh. All results of v3 (and v4)
depend on this alignment.

**Publication times** (`PUB_TIME`) are also localised as Athens. They are
consistent with Athens time: IDA2 23:23–23:34 and IDA3 11:22–11:27 local are
about 20 min after the SIDC gate closures (22:00 and 10:00 CET = 23:00 and
11:00 Athens); read as CET they would be 80 min late. Not verified against a
HEnEx statement.

**Weather time stamps (extra, to check).** Over all train and test days the
irradiance-weighted mean time is later than solar noon (pvlib, Athens): actual
GHI +23.5/+24.7 min (test/train), short-lead forecast +24.2/+23.7,
`_previous_day1` +23.7/+23.6, `_previous_day2` +21.9/+22.8, against −2.9/−0.5
min for the pvlib clear-sky GHI on the same time stamps. This is consistent
with hourly values that average the preceding hour being treated as
instantaneous values at the end of that hour (expected offset 30 min). The
Open-Meteo convention has not been checked against its documentation here.
This shift and the 0.3 shift point the same way, so PV leads prices by
about 1.4 h in the dataset.

## 0.4 DST days — done

`_build_day_table` keeps only calendar days with exactly 96 steps
([pv_bess_greece_env.py:354-365](../pv_bess_gym/envs/pv_bess_greece_env.py)).
Dataset: 2025-03-30 has 92 steps, 2025-10-26 has 100; both are excluded
(363 days). Split sizes: train 169, validation 61, test 91, held-out 42.

## 0.5 Commercial position limits — issue

The environment clips the DAM schedule and the intraday correction
*separately* to ±`max(P_exp, P_imp)` = ±373 kW
([pv_bess_greece_env.py:1111-1124, 1221-1222, 1270](../pv_bess_gym/envs/pv_bess_greece_env.py)).
`q = x + Y` is not bounded, so `q ∈ [−746, 746]` kW is possible. The physical
export is capped by the grid model; the commercial position is not.

T1 test (`part0_tables.json`, `position_limits`):

| Controller | Steps `q > P_exp` | Energy above `P_exp` [kWh/day] | Steps `q < −P_imp` | max `q` [kW] |
|---|---:|---:|---:|---:|
| DA-ONLY | 0.00% | 0.0 | 0.00% | 369.3 |
| DA-OPT-RBC / -RT-MPC | 1.00% | 12.1 | 0.00% | 566.3 |
| DA-OPT-ID-MPC / -MPC | 11.45% | 416.7 | 0.00% | 742.3 |
| PPO (5 seeds) | 9.49% | 221.9 | 0.00% | 742.3 |
| SAC (5 seeds) | 22.27% | 715.5 | 0.00% | 742.3 |
| PPO-Lagrangian (5 seeds) | 17.22% | 503.8 | 0.00% | 742.3 |

`x ≥ 0` on every step (the D-1 QP has `schedule ≥ 0`).

## 0.6 Oracle feasible set vs the D-1 QP — done (list)

Constraints/terms of the D-1 QP ([day_ahead_baselines.py:280-343](../pv_bess_gym/agents/day_ahead_baselines.py))
that the oracle MILPs ([oracle.py:163-255](../pv_bess_gym/oracle.py)) do not have:

1. `x ≥ 0` (oracle: `x ∈ [−373, 373]`).
2. `r± ≤ 0.5 P_pcs` = 186.5 kW (`reserve_max_fraction`; oracle: `r± ≤ P_pcs`).
3. `r⁺ + r⁻ ≤ P_pcs` (oracle: none).
4. `p_dis + r⁺ ≤ P_pcs`, `p_ch + r⁻ ≤ P_pcs` as hard constraints (oracle: allowed,
   penalised as in the environment).
5. Energy headroom with buffer `E_buf = 0.08 E_max` = 86 kWh and sustain 0.5 h:
   `e ≥ E_min + E_buf + (0.5/η) r⁺`, `e ≤ E_max − E_buf − 0.5 η r⁻` (oracle: none).
6. Balanced schedule `x = planned metered injection` (zero planned imbalance)
   (ORACLE: none; ORACLE-DELIVER: zero imbalance including activation, which is
   a different, step-wise delivery constraint).
7. Objective terms: throughput penalty 2.5 EUR/MWh, terminal SOC penalty
   (target 0.55, weight 12), unserved-auxiliary penalty 1000 EUR/MWh (oracle:
   none of these; it maximises the realised reward).
8. D-1 information: the QP sees forecast PV and forecast DAM prices; the oracle
   sees realised values and can choose intraday corrections.

Mean committed capacity on T1 test:

| | r⁺ [kW] | r⁻ [kW] | days with r⁻ > 0 |
|---|---:|---:|---:|
| DA-ONLY (all D-1 baselines and RL share this plan) | 186.50 | 161.64 | 95.6% |
| ORACLE, all 10 seeds (910 solves) | 193.59 | 36.09 | 19.1% |
| ORACLE, seed 0 (91 solves) | 195.54 | 45.15 | 22.0% |
| ORACLE-DELIVER, seed 0 (91 solves) | 322.50 | 228.41 | 82.4% |

`r⁺` of the D-1 QP equals the 0.5 `P_pcs` cap on every test day. ORACLE
commits `r⁺` above that cap in 55.1% of episodes and `r⁺ + r⁻ > P_pcs` in
9.1%. ORACLE-DELIVER commits on average `r⁺ + r⁻` = 550.9 kW, well above
`P_pcs` = 373 kW; constraints 2, 3 and 5 are clearly binding for a
like-for-like comparison (input for A5). The DA-ONLY row holds for every
controller (all share the D-1 plan; the step logs give 186.50 / 161.64 kW for
each of the eight controllers).

Source: `results/rev1/part0/oracle_reserve_{oracle,deliver}.csv`
(`oracle_reserve.py`, protocol-v3 solver settings). Reproduction check against
`results/v3/oracle_T1-Nominal.csv`: ORACLE bounds max |Δ| = 0.0115 EUR
(910 solves). ORACLE-DELIVER bounds max |Δ| = 9.53 EUR, mean Δ −0.11 EUR;
incumbents max |Δ| = 7.46 EUR (91 solves). The difference comes from the
120 s time limit: where HiGHS stops at the limit, the bound and incumbent
depend on how far the solver got, which depends on machine load. **ORACLE-DELIVER
values are not bit-reproducible** (31% of T1 solves hit the limit, see 0.8).
Compute: ORACLE 1.72 CPU-h of solver time (7,018 s user, 10.3 min wall);
ORACLE-DELIVER 1.24 CPU-h (4,599 s user, 7.0 min wall).

## 0.7 Upper-bound argument — done

- `η̄` in the oracle: `clip(eta_nom, 0.85, 0.995)` = **0.97**
  ([oracle.py:134](../pv_bess_gym/oracle.py)). The environment efficiency is
  `0.97 − 0.02 |p|/P_pcs` ∈ [0.95, 0.97] ([battery.py `eta`](../pv_bess_gym/models/battery.py)),
  never above `η̄`.
- Self-discharge order. Environment: `e1 = sd · (e0 − p_net dt/η)` (discharge) or
  `sd · (e0 + |p_net| dt η)` (charge), then clip ([battery.py `step`](../pv_bess_gym/models/battery.py)).
  Oracle: `e[t+1] = sd·e[t] − sd·dt/η̄·dis + sd·dt·η̄·ch − w + gain`
  ([oracle.py:217-219](../pv_bess_gym/oracle.py)). **Both apply self-discharge
  to `e_k` plus the step's flow**; the manuscript's statement that the
  environment applies it to `e_k` only is wrong (it describes the version-4
  D-1 QP and the real-time MPC, [day_ahead_baselines.py:301, 565](../pv_bess_gym/agents/day_ahead_baselines.py)).
- Feasibility of every environment trajectory: with the same order and
  `η ≤ η̄`, the environment's energy after a discharge is `sd(e0 − p dt/η) ≤ sd(e0 − p dt/η̄)`
  and after a charge `sd(e0 + c dt η) ≤ sd(e0 + c dt η̄)`; the difference is a
  non-negative `w`. The clip at `E_max` is a further `w ≥ 0`; the floor
  behaviour (unserved auxiliary load, no self-discharge below `E_min`) is
  modelled with the `floor`/`gain`/`unserved` variables. Power, curtailment,
  grid, activation and penalty terms are supersets of the environment
  (module docstring, [oracle.py:9-32](../pv_bess_gym/oracle.py)).
  **No case found where the argument fails; worst-case gap: none.**
- Empirical check, all five tasks, 194,000 episodes (`results/v3/benchmark_*.csv`):
  0 episodes with profit > `U`; max(profit − `U`) = −336.6 EUR (T4) to −441.4 EUR (T3);
  max(`U_rep` − `U`) = −0.20 to −0.22 EUR.

## 0.8 ORACLE-DELIVER bound vs incumbent — done

Test split, 910 episodes per task; mean daily value with 95% day-bootstrap CI
(5,000 resamples, seed 0, same routine as `summarise_journal.day_bootstrap`).
T1 and T2 share one oracle run.

| Task | `U_D` (bound) | `U_D` incumbent | median gap | p90 gap | max gap | gap > 5% | gap > 1% | solves ≥ 120 s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| T1/T2 | 575.11 [547.42, 603.84] | 574.26 [546.52, 602.96] | 0.00010 | 0.00442 | 0.07125 | 3 | 21 | 31.0% |
| T3 | 573.35 [545.52, 601.89] | 571.77 [543.79, 600.38] | 0.00010 | 0.00707 | 0.13208 | 6 | 70 | 33.1% |
| T4 | 567.75 [540.30, 596.14] | 566.52 [539.09, 594.87] | 0.00010 | 0.00539 | 0.11902 | 6 | 42 | 35.7% |
| T5 | 565.94 [538.34, 594.21] | 564.29 [536.62, 592.62] | 0.00010 | 0.00653 | 0.18308 | 9 | 62 | 37.1% |

Mean bound − incumbent: 0.84 (T1/T2), 1.58 (T3), 1.24 (T4), 1.65 (T5) EUR/day.
For comparison, ORACLE (`U`): max gap 0.00042–0.00266, none above 5%.

## 0.9 Table 14 timings — issue (manuscript)

Source: `train.json` of each main run (25 per algorithm). Both numbers in
`results/v3/compute.json` are computed correctly
([measure_compute.py:105-113](../scripts/measure_compute.py)):
`mean_train_minutes_per_seed` = mean of `train_seconds` / 60;
`mean_steps_per_second` = mean over runs of `300000 / train_seconds`. The
mean of ratios is not the ratio of means, and a few runs were much slower:

| Algorithm | mean min | median min | min–max s | mean of per-run steps/s | 300k / mean time | 300k / median time | runs > 1.5× median |
|---|---:|---:|---|---:|---:|---:|---:|
| PPO | 2.98 | 2.97 | 174.6–183.4 | 1680.0 | 1679.6 | 1684.4 | 0 |
| SAC | 35.59 | 26.95 | 1603.4–4223.7 | 162.9 | 140.5 | 185.5 | 5 (all T5-Combined) |
| PPO-Lagrangian | 7.93 | 4.94 | 289.0–2912.6 | 940.0 | 630.6 | 1012.8 | 2 (both T5-Combined) |

What each measures: wall-clock seconds of `model.learn` (SB3,
[train_sb3.py:70-72](../scripts/rl/train_sb3.py)) or of `omnisafe.Agent(...)` plus
`agent.learn()` (OmniSafe, [train_ppol.py:68-71](../scripts/rl/train_ppol.py)),
including checkpoint saving every 30,000 steps, **excluding** checkpoint
evaluation (done afterwards by `eval_rl.py`). The slow T5 runs ran in parallel
with other jobs; PPO on T5 was not slow, so the environment is not the cause,
but the cause (CPU contention) was not recorded. Recommended for Table 14:
report the median, or state "mean time; mean of per-run rates".

## 0.10 Observations per controller class — done

Observation dict ([pv_bess_greece_env.py:1308-1371](../pv_bess_gym/envs/pv_bess_greece_env.py)),
protocol v3 (information filter on).

**RL** (`flatten_observation`, [wrappers.py](../pv_bess_gym/rl/wrappers.py)): 97 values =
24 `physical` + 8 × 9 `forecast` + 1 intraday-due flag, each divided by a fixed scale.

`physical` (step `t`): 0 SOC; 1 energy [kWh]; 2 **realised PV of step t**;
3 PV forecast for t; 4 GHI, 5 air temperature, 6 wind (measured, step t);
7 DAM price (realised, after commitment); 8 intraday price (newest *published*
auction, else DAM); 9 imbalance "price" (= DAM price; imbalance is never
published in-day); 10–11 aFRR capacity prices up/down; 12–13 **aFRR activation
ratios of step t**; 14 `x_t`; 15 `Y_t`; 16 `x_t + Y_t`; 17 steps to next
intraday decision; 18 steps left in aFRR block; 19–20 time of day sin/cos;
21 `t/95`; 22 `r⁺`; 23 `r⁻`.

`forecast` rows `t+1 … t+8`: PV forecast, GHI forecast, temperature forecast,
DAM price, published intraday price, imbalance stand-in (= DAM), capacity
prices up/down, `x + Y`.

So RL **sees** the D-1 schedule and its intraday position (current step and
8 steps ahead), the reserve commitment `r±`, the 8-step forecast matrix and
the published intraday price. It does **not** see the full-day
`day_ahead_context`, the D-1 planned battery trajectory, or forecasts beyond
8 steps. Note: the realised PV (index 2) and the activation ratios (12–13) of
the step being decided are observable before the action ("measured at the
current step", `PUBLICATION_RULES["measurement"]`); the rule and the
real-time MPC also use the realised PV of step t (`physical[2]`).

**Baselines** receive the full dict. D-1 QP (step 0, before commitment): PV
forecast `_previous_day2` and naive DAM forecast (`day_ahead_context`
columns 0–1), aFRR capacity prices (columns 4–5, ISP1, published D-1).
Intraday rule: PV forecast of the target block and `x`. Intraday MPC: PV
forecast, `x`, published intraday price, DAM price as imbalance stand-in, all
for the target block. Real-time rule: SOC, realised PV of step t, DAM price,
`x + Y`, `r±` (own D-1 plan). Real-time MPC: SOC, realised PV of step t,
forecast PV, DAM stand-in and `x + Y` for 7 further steps, `r±`.

## 0.11 Statistics — done

- **IQM and CI** ([summarise_journal.py `score_aggregates`](../scripts/summarise_journal.py)):
  `rliable` 1.2.0 `metrics.aggregate_iqm` (25% trimmed mean over all episodes)
  with `rly.get_interval_estimates(..., method="percentile", reps=5000,
  RandomState(0))`. The score matrix is (episodes × 1 task), so the
  "stratified" bootstrap has one stratum and resamples **episodes**
  (day × activation seed, × RL seed) i.i.d.; days are not kept together.
  Mean and median use the same bootstrap. Episodes with `U − J_L < 1 EUR` are
  dropped before.
- **CVaR_5%** (`cvar`): mean of the lowest `ceil(0.05 n)` episode profits
  (episodes, not days); no CI.
- **Day bootstrap for profits** (`day_bootstrap`): per-day mean over that day's
  episodes, resample days with replacement (5,000 reps, seed 0), 95% percentile
  interval.
- **Paired 2×2 attribution**: per (day, activation seed), contrasts
  `Δ_ID`, `Δ_RT` and interaction on scores of the four DA-OPT controllers
  (episodes valid for all four), then `day_bootstrap` of the contrast.
- **Activation seeds**: `activation_series` draws from
  `SeedSequence([activation_seed, start_idx])`, independent of the controller
  ([pv_bess_greece_env.py:391-405](../pv_bess_gym/envs/pv_bess_greece_env.py)); the
  oracles call the same function ([oracle.py:155](../pv_bess_gym/oracle.py)).
  Shared: confirmed.

## Extra: reproducibility of RL evaluation — issue (small)

Replaying the selected RL checkpoints on T1 test in chunks of 8 days gives
profits that differ from `results/v3/benchmark_T1-Nominal.csv` in about 55%
of episodes: mean |Δ| 0.03–0.08 EUR per seed, max 8.98 EUR (SAC seed 3); mean
daily profit PPO 278.791 vs 278.805, SAC 271.941 vs 271.949 EUR. Replaying PPO
seed 0 on all 91 test days in one process, as `eval_rl.py` does, reproduces
the file exactly (max |Δ| 5.7e-14). Cause: the D-1 QP is compiled once and
solved with `warm_start=True` ([day_ahead_baselines.py:219](../pv_bess_gym/agents/day_ahead_baselines.py)),
so its solution depends at solver tolerance on the previous solve in the
process; RL policies amplify these differences. Baselines are run in the same
8-day chunks by `run_journal.py` and reproduce exactly.

---

## Open items for Part A (proposal, not started)

- A1 applies (0.1 confirmed PV-only).
- A2 applies (0.5: up to 24% of steps above `P_exp`).
- **A3 is larger than the brief assumes**: not only IDA3 but every market
  series (DAM, IDA1–3, imbalance, aFRR capacity) needs the CET/CEST → Athens
  conversion; the dataset, publication sidecars, every oracle and every
  controller must be re-run (all tasks, including RL evaluation; RL retraining
  is the author's decision, since the training data change too).
- The weather time-stamp question should be settled before A3 is re-run, so
  the data are rebuilt only once.
- A4 does not apply (0.7).
- A5, A6 as in the brief. For A6, 0.2 gives the share of trades with a known price.
- Small: make the D-1 QP solve independent of solve order (e.g. `warm_start=False`)
  if bit-identical replays in any chunking are wanted.

Author decisions needed before Part A: (1) the scope of A3 (all market data,
full re-run of every task); (2) RL: retrain or re-evaluate after A3; (3) the
weather time-stamp convention; (4) the compute budget for a full re-run, which
will be estimated before starting.

---

# Part A — corrections implemented (stopped for author approval)

Author instruction (2026-11-18): fix the time alignment in the model. Implemented
behind flags; protocol `configs/journal.yaml` is now **rev1**, v3 is frozen as
`configs/journal_v3.yaml` (re-checked: 5 baselines × 8 test days × 10 seeds
reproduce `results/v3` to 5.7e-14 EUR). CHANGELOG has one row per flag.

| Fix | Flag (rev1 value) | Where |
|---|---|---|
| A3 market clock | `market_clock: cet` — all market columns and sidecar auction prices moved 1 h later at load | `pv_bess_gym/data/time_alignment.py`, `envs/factory.py` |
| A3 irradiance stamps | `radiation_stamp: interval_centre` — GHI (actual, short-lead, `_previous_day1/2`) moved 30 min earlier; Open-Meteo documents radiation as "average of the preceding hour", temperature and wind as instantaneous | same |
| A1 intraday reference | `id_reference: d1_plan` | `agents/day_ahead_baselines.py` |
| A2 position limit | `id_position_limit: clip` — environment, intraday MPC **and both oracles** (otherwise the oracle plan used positions the environment clips: `U_rep` fell to 1174.7 with `U` 1715.8) | `envs/pv_bess_greece_env.py`, `oracle.py` |
| Oracle robustness | one retry with 4× time limit when HiGHS returns no incumbent (needed once in 910 rev1 T1 solves; never on v3) | `oracle.py`, column `retries` |

Not applied: A4 (not needed, 0.7). A5, A6 not started.

Checks after the change: rev1 drops 2025-01-01 (no first market hour; 362 days,
train 168); IDA3 covers local 13:00–24:00 (plus 00:00–01:00, which belongs to
the previous trading day's IDA3); test-split GHI centroid − solar noon moves
from +23.5 to −6.5 min (pvlib clear sky −2.9 min).

**Tests** (`make test`, before the oracle change): 98 passed, 1 failed;
`tests/test_rev1.py` 11 passed (incl. the oracle position test, run separately).
The failure: `test_R4_balanced_d1_plan_delivered_exactly_has_zero_imbalance`
under rev1, 2025-11-16 steps 43–56, imbalance 0.002–0.010 kWh per step (tolerance
1e-4). Cause: on that day the R4 D-1 QP disposes of surplus PV by simultaneous
import and export of about 0.8 kW (wasting it through the 1 % metering loss),
which the plant cannot do. Pre-existing QP property exposed by the rev1 data;
not fixed, test not loosened. Author decision needed.

## Effect on T1-Nominal, test split (ORACLE and baselines only)

`results/rev1/impact_T1.json` (`results/rev1/impact_T1.py`). 910 episodes per
controller. Profit: mean EUR/day [95% day-bootstrap CI]; S: IQM [95% episode
bootstrap], 5,000 resamples, seed 0. ORACLE-DELIVER (S_D) and RL not re-run.

| | v3 | rev1 |
|---|---|---|
| U (ORACLE bound) | 1741.8 | 1231.7 |
| U_rep | 1706.8 | 1194.0 |
| J_L (DA-ONLY) | 390.6 | 373.7 |
| max MIP gap (ORACLE) | 0.00042 | 0.09670 (27 of 910 solves ≥ 120 s) |

| Controller | v3 profit | rev1 profit | v3 S IQM | rev1 S IQM | rev1 scheduled / metered kWh/day | rev1 imbalance EUR/day |
|---|---|---|---|---|---|---|
| DA-ONLY | 390.6 [367.8, 414.0] | 373.7 [352.1, 396.2] | 0 | 0 | 1702.2 / 1791.0 | +6.9 |
| DA-OPT-RBC | 325.9 [308.2, 344.1] | 374.5 [353.0, 396.8] | −0.0486 [−0.0509, −0.0462] | +0.0005 [−0.0002, +0.0011] | 1705.2 / 1793.5 | +6.3 |
| DA-OPT-RT-MPC | 294.5 [277.6, 311.9] | 365.4 [343.3, 388.4] | −0.0725 [−0.0752, −0.0698] | −0.0149 [−0.0166, −0.0133] | 1705.2 / 1705.5 | −3.3 |
| DA-OPT-ID-MPC | 341.6 [315.7, 369.2] | 370.0 [347.5, 392.6] | −0.0261 [−0.0305, −0.0219] | **+0.0030 [+0.0011, +0.0049]** | 2288.2 / 1818.9 | −107.8 |
| DA-OPT-MPC | 345.1 [319.1, 372.4] | 370.2 [348.0, 393.9] | −0.0266 [−0.0302, −0.0231] | −0.0050 [−0.0078, −0.0023] | 2288.2 / 1762.5 | −108.1 |

Reading (no attribution to the individual fixes; that would need one run per flag):

- J_L falls by 16.9 EUR/day (−4.3%); U falls by 510 EUR/day (−29%), mostly a
  consequence of the position bound in the oracle (U without it, same data and
  baselines: 1715.8).
- **Over-selling after A1:** the rule no longer over-sells (scheduled 1705.2 vs
  metered 1793.5 kWh/day, like DA-ONLY). The intraday MPC still does:
  scheduled 2288.2 vs metered 1762.5–1818.9 kWh/day, imbalance −108 EUR/day.
  Its remaining driver is the price term (0.1): it sells whenever the published
  intraday price exceeds its imbalance weight max(140, |DAM|).
- **The finding "no controller beats DA-ONLY" no longer holds for the IQM:**
  DA-OPT-ID-MPC has S IQM +0.0030 with a CI above zero, while its mean profit
  (370.0) is below DA-ONLY (373.7) with overlapping CIs. DA-OPT-RBC is
  indistinguishable from DA-ONLY.
- The ORACLE MIP gap grows from 0.0004 to up to 0.097 with the position bound;
  the reported U is the dual bound, so it stays a valid bound, slightly looser
  on those 27 episodes.

Compute for Part A so far: 5.2 + 1.9 (first two oracle runs, discarded) +
6.4 (final oracle) + 1.3 (baselines) + tests ≈ 15 CPU-h.

## Decisions needed before the full rev1 run

1. Approve Part A and the full re-run (all 5 tasks: ORACLE, ORACLE-DELIVER on
   the test split, baselines, RL retraining and evaluation). The rev1 oracle is
   slower (T1 test: 6,383 s user, mean 22.5 s per solve, max 1,064.6 s);
   ORACLE-DELIVER with the position bound has not been timed.
2. The R4 test failure (QP disposal by simultaneous import/export): fix in the
   QP (behind a flag) or document.
3. Whether the paper keeps "no controller beats DA-ONLY" pending the full run.

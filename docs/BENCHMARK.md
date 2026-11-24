# Benchmark protocol (journal protocol rev1)

The protocol is defined in [`configs/journal.yaml`](../configs/journal.yaml)
(name `pvbess-greece-2025-journal-rev1`) and the five task files in
[`configs/tasks/`](../configs/tasks). Earlier protocols are frozen as
[`configs/journal_v3.yaml`](../configs/journal_v3.yaml) and
[`configs/journal_v4.yaml`](../configs/journal_v4.yaml); select one with
`PVBESS_PROTOCOL=<file>` (or `make ... PROTOCOL=<file>`). Every behaviour
change between protocols has its own flag, listed in [CHANGELOG.md](../CHANGELOG.md).

## Plant and decision sequence

A 550 kWp PV plant with a 373 kW / 1,075 kWh battery behind a 373 kW grid
connection (`example_data/station_athens.json`) operates for one delivery day
(00:00–24:00 Europe/Athens) per episode, in 15-minute steps, from SOC 0.5:

1. **D-1:** a 96-step day-ahead (DAM) schedule and one upward and one downward
   aFRR capacity for the day.
2. **Intraday:** every 60 minutes, one energy correction for the 60-minute
   block that starts after the gate-closure lead (60 or 120 min, per task).
   The correction is clipped so that the commercial position
   `x + Y` stays within the grid limits (rev1 `id_position_limit: clip`).
3. **Real time:** every 15 minutes, the battery setpoint.

The reward is the settled profit of the step in EUR (settlement v2): DAM,
intraday, imbalance on the metered injection (1 % metering loss), aFRR
capacity and activation energy, minus degradation and penalties. Settlement
always uses realised prices.

## Information (what a controller may observe)

With `info_filter: true`, a value is observable only after its publication
time (HEnEx results `PUB_TIME`, IPTO ISP1 run time; rules in
`pv_bess_gym/envs/information.py`). The D-1 layer sees a naive previous-day
DAM price forecast and the Open-Meteo Previous Runs weather forecast issued at
least 48 h before the valid time; after the commitment the intraday and
real-time layers see the short-lead forecast up to 3 h ahead and the 24 h-lead
forecast beyond. Intraday prices are the newest *published* auction
(IDA1 → IDA2 → IDA3); imbalance prices are never observable in-day.

Market data are aligned to Athens time at load (rev1 `market_clock: cet`: the
HEnEx and IPTO files are in CET/CEST wall clock), and hourly Open-Meteo
irradiance is placed at the centre of its averaging hour
(`radiation_stamp: interval_centre`). See [DATA.md](DATA.md).

## Tasks

| Task | κ (forecast residual scaling) | Gate lead | Degradation cost c_thr |
|---|---|---|---|
| T1-Nominal | 1.0 | 60 min | 0 |
| T2-Forecast | 2.0 | 60 min | 0 |
| T3-Gate | 1.0 | 120 min | 0 |
| T4-Ageing | 1.0 | 60 min | 5.7 EUR/MWh |
| T5-Combined | 2.0 | 120 min | 5.7 EUR/MWh |

κ scales the residual (forecast − actual) of every weather series a layer
receives; κ = 1 is the archived forecast. All tasks use the forecast DAM price
at D-1. The tasks are registered as `PVBESS-<task>-v2` for `gymnasium.make`.

## Data split

| Split | Delivery dates | Use |
|---|---|---|
| train | 2025-01-02 – 2025-07-31, without the held-out weeks | RL training |
| validation | 2025-08-01 – 2025-09-30 | RL checkpoint selection, PPOL cost budget |
| test | 2025-10-01 – 2025-12-31 | reported results |
| held-out weeks | ISO weeks 5, 10, 15, 20, 25, 30 of 2025 (inside the train range) | within-regime comparison |

DST-change days (2025-03-30, 2025-10-26) are not delivery days. Under rev1,
2025-01-01 is not a delivery day either: its first market hour
(23:00–24:00 CET of 2024-12-31) is not in the dataset. Every (day, activation
seed) pair is one episode; activation seeds 0–9 are shared by all controllers
and the oracles.

## Bounds and scores

- **ORACLE**: perfect-foresight MILP (HiGHS, 120 s limit, MIP gap 1e-4) with
  every action the environment allows, under a relaxation of the plant
  physics. Its proven dual bound `U` is an upper bound on every policy's profit
  for the same realisation.
- **ORACLE-DELIVER**: the same MILP constrained to meter its own schedule at
  every step; dual bound `U_D`. It bounds every policy that delivers its
  schedule, not policies that carry imbalance. Under rev1 it is solved on the
  test split for T1-Nominal and T2-Forecast (they share one oracle).
- **DA-ONLY** (lower anchor): the shared D-1 plan, no intraday trades,
  rule-based real-time tracking; profit `J_L`.

Normalised scores per episode: `S = (J − J_L)/(U − J_L)` (primary) and
`S_D = (J − J_L)/(U_D − J_L)`. Episodes with a range below 1 EUR are excluded
and counted. Task scores are interquartile means over episodes with 95%
bootstrap CIs (`rliable`, 5,000 resamples); daily profits use a day-level
bootstrap. Implementation: `scripts/summarise_journal.py`.

## Baselines

All controllers share the D-1 optimisation (convex QP, CLARABEL) of
`pv_bess_gym/agents/day_ahead_baselines.py`:

| Controller | Intraday layer | Real-time layer |
|---|---|---|
| DA-ONLY | none | rule |
| DA-OPT-RBC | rule | rule |
| DA-OPT-RT-MPC | rule | MPC |
| DA-OPT-ID-MPC | MPC | rule |
| DA-OPT-MPC | MPC | MPC |
| DA-OPT-PPO / -SAC / -PPOL | learned (Stable-Baselines3 PPO, SAC; OmniSafe PPO-Lagrangian) | learned |

Under rev1 the intraday rule and MPC expect to deliver the D-1 planned metered
injection updated by the PV forecast change (`id_reference: d1_plan`). RL
agents get 300,000 environment steps, a learning rate chosen on T1 from
{1e-4, 3e-4, 1e-3} (one seed), five training seeds, and the checkpoint with the
best validation score; the PPO-Lagrangian cost budget is the mean validation
cost of DA-OPT-MPC.

## Reproducing

`make benchmark` runs every stage of the active protocol; [`results/COMMANDS.md`](../results/COMMANDS.md)
maps each result file to its command. Per-job files go to `results/<subdir>/parts/`
and are skipped when present, so an interrupted run resumes.

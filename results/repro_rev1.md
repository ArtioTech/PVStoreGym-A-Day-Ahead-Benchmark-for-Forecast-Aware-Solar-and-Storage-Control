# Reproducibility report, protocol rev1

## Code and environment

| Item | Value |
|---|---|
| Code | commit `e854535` plus the uncommitted rev1 changes in the working tree (`compute.json` records `dirty: true`). The author commits and pushes; record the new commit hash here afterwards. |
| Package version | `pv_bess_gym` 2.1.0 |
| Protocol | `configs/journal.yaml` (`pvbess-greece-2025-journal-rev1`), `PVBESS_PROTOCOL` unset or set to that file |
| Main environment `.venv` | Python 3.14.10; numpy 2.4.2, scipy 1.17.1, pandas 3.0.1, gymnasium 1.2.3, cvxpy 1.8.1, clarabel 0.11.1, highspy 1.13.1, pvlib 0.15.0, stable-baselines3 2.9.2, torch 2.15.0 (`requirements.lock`) |
| OmniSafe environment `.venv-omnisafe` (PPO-Lagrangian, rliable statistics) | Python 3.9.6; omnisafe 0.5.0, rliable 1.2.0, torch 2.8.0, numpy 1.26.4, gymnasium 0.28.1, pandas 2.0.3, scipy 1.13.1, cvxpy 1.7.5 (`requirements-omnisafe.lock`) |
| Solvers | QP: CLARABEL via CVXPY. MILP: HiGHS via `scipy.optimize.milp` (120 s limit, relative gap 1e-4, one retry at 480 s without incumbent) |
| Hardware | Apple M5 Pro, 15 cores, 48 GB, macOS, CPU only; 11–12 parallel jobs |
| Seeds | activation seeds 0–9 (all controllers and oracles); RL training seeds 0–4; tuning seed 0; bootstrap seed 0 |

## Commands, exit status and wall time

The full run was `results/rev1/run_full.sh` (one stage per line; logs in
`results/rev1/logs/`, exit status and seconds in `results/rev1/logs/stages.tsv`).
The ORACLE stage was started separately before it (`results/rev1/run_oracle.log`).
All commands with `PVBESS_PROTOCOL=configs/journal.yaml`.

| Stage | Command | Exit | Wall time |
|---|---|---|---|
| ORACLE | `.venv/bin/python scripts/run_journal.py oracle --tasks T1-Nominal T2-Forecast T3-Gate T4-Ageing T5-Combined --jobs 11` | 0 | 2 h 28 min (73,695 s user) |
| ORACLE (check, all parts present) | `… run_journal.py oracle --tasks … --jobs 12` | 0 | 3 s |
| Baselines | `.venv/bin/python scripts/run_journal.py baselines --tasks … --jobs 12` | 0 | 68 min |
| RL tuning (T1) | `.venv/bin/python scripts/rl/run_all.py tune --jobs 12 --python .venv/bin/python --python-omnisafe .venv-omnisafe/bin/python` | 0 | 19 min |
| RL training and evaluation | `.venv/bin/python scripts/rl/run_all.py main --tasks … --jobs 12 --python … --python-omnisafe …` | 0 | 7 h 12 min, including idle-sleep pauses (see below; the T3-Gate SAC seed-0 run alone recorded 5.6 h of them) |
| RL collection | `.venv/bin/python scripts/rl/collect_rl.py` | 0 | 2 s |
| ORACLE-DELIVER (T1/T2, test) | `.venv/bin/python scripts/run_journal.py oracle_deliver --tasks T1-Nominal T2-Forecast --jobs 12` | 0 | 1 h 55 min |
| Aggregate | `.venv/bin/python scripts/run_journal.py aggregate --tasks …` | 0 | 6 s |
| Summaries | `.venv-omnisafe/bin/python scripts/summarise_journal.py --tasks …` | 0 | 44 s |
| Compute | `.venv/bin/python scripts/measure_compute.py` | 0 | 33 s |
| Figures | `make figures` | 0 | < 1 min |
| Tables | `.venv/bin/python results/rev1/make_tables.py > results/rev1/tables_rev1.md` | 0 | < 1 min |
| DAM forecast error | `.venv/bin/python scripts/data/da_forecaster_error.py` → `results/rev1/da_forecaster_error.json` | 0 | < 1 min |
| Settlement conformance | `.venv/bin/python scripts/validation/validate_settlement.py` → `results/rev1/conformance.json` | 0 | ~1 min |
| Tests | `make test` → `results/rev1/tests.json` | 2 (one failure, below) | 7 min 28 s |

`make benchmark` runs the same stages in sequence (with `--jobs` at the
script defaults).

**Idle-sleep pauses.** From about 02:30 to 08:30 the machine idle-slept
repeatedly: every process stopped at the same moments, about every 16 minutes,
for about 15 minutes each time (per-episode time stamps in the RL monitor files).
This affected only wall time, not results: training is driven by environment
steps and seeds, not by the clock. Sleep was blocked for the rest of the run.
Pause-free training times are in `results/rev1/rl_training/train_time_sleep_corrected.csv`.

## Tests

`make test`: **99 passed, 1 failed** (`results/rev1/tests.json`, `results/rev1/make_test.log`).

The failure is `tests/test_realism_v3.py::test_R4_balanced_d1_plan_delivered_exactly_has_zero_imbalance`
under rev1. On 2025-11-16, steps 43–56, the D-1 QP disposes of surplus PV by
simultaneous import and export of about 0.8 kW, which gives 0.002–0.010 kWh of
imbalance per step against a 1e-4 tolerance. It is a property of the D-1 QP
exposed by the rev1 data. It is not fixed and the test is not loosened
(author decision, `results/paper_inputs_rev1.md`).

## Checks

- **v3 still reproduces.** With `PVBESS_PROTOCOL=configs/journal_v3.yaml`,
  5 baselines × 8 test days × 10 seeds match the v3 results to 5.7e-14 EUR
  (`results/status_rev1.md`, Part A).
- **Settlement conformance.** 5 days × 5 settlement legs recomputed from the raw
  HEnEx/IPTO files: largest daily deviation 0.0000 EUR under rev1 (raw files
  read as CET/CEST) and under v3 (raw files read as Athens time).
- **Bounds.** No episode on any task has a profit above U (test `Oracle >= every
  baseline`, and Table A).
- **ORACLE-DELIVER** re-solves that hit the 120 s limit are not bit-reproducible
  (up to 9.53 EUR per episode in the Part 0 re-solve); 36.7% of the rev1 solves hit it.
- **RL evaluation** depends on the order of D-1 QP solves within one process
  (CLARABEL warm start), up to 8.98 EUR per episode and ≤ 0.014 EUR/day in the
  means (Part 0); a sequential replay reproduces exactly.

## Table 13 (commands in the manuscript)

`main.tex` is not in the repository, so Table 13 could not be checked
line by line. `results/COMMANDS.md` lists the rev1 commands and is consistent with
the table above; Table 13 should be copied from it.

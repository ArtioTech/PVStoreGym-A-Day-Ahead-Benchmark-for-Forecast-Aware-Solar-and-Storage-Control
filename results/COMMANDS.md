# Commands that regenerate the results

Two Python environments are used (see `docs/INSTALL.md`):

- `.venv` — Python 3.14, `requirements.lock`: simulator, oracle, optimisation
  baselines, PPO and SAC (Stable-Baselines3), figures.
- `.venv-omnisafe` — Python 3.9, `requirements-omnisafe.lock`: PPO-Lagrangian
  (OmniSafe 0.5.0) and the `rliable` statistics.

Identical episodes give bit-identical profits in both environments.
`make` uses them by default; override with `PYTHON=` and `PYTHON_OMNISAFE=`.

Every results file has a `meta` block (or a `.meta.json` next to it) with the
git commit, package and solver versions, seeds and wall-clock time.

**Protocol versions.** `make` reads the protocol from `PROTOCOL` (exported as
`PVBESS_PROTOCOL`):

- `configs/journal.yaml` (default, **rev1**, final technical revision): v3 plus
  the corrections A1 (`id_reference: d1_plan`), A2 (`id_position_limit: clip`)
  and A3 (`market_clock: cet`, `radiation_stamp: interval_centre`, see
  `pv_bess_gym/data/time_alignment.py`). Results in `results/rev1/`, figures
  in `figures/rev1/`. Part 0 diagnostics: `results/rev1/part0/`.
- `configs/journal_v3.yaml` (**v3**, realism fixes R1–R6, frozen): writes to
  `results/v3/` and `figures/v3/` when re-run (those results are not kept in the repository). Run with `make benchmark
  PROTOCOL=configs/journal_v3.yaml`. Adds the step `run_journal.py
  oracle_deliver` (ORACLE-DELIVER, test split only, `oracle.deliver_splits`),
  the bound `U_D` and the score `S_D`. The diagnostics are
  `scripts/diagnostics/schedule_vs_metered.py`,
  `scripts/diagnostics/forecast_error_by_layer.py` and
  `scripts/diagnostics/t1_da_only_analysis.py`, which write to
  `results/v3/diagnostics/` when run under v3. Weather leads come from
  `scripts/data/fetch_open_meteo_previous_runs.py`.
- `configs/journal_v4.yaml` (**v4**, as submitted before the realism fixes):
  writes to `results/` and `figures/` when re-run (not kept). Run with `make benchmark
  PROTOCOL=configs/journal_v4.yaml`. On T1 it reproduced all 9,700 baseline
  episodes and PPO seed 0 bit for bit.

The paths below are those of rev1. Other protocols write to their own
`results_subdir` (`results/v3/`, `figures/v3/`; v4: `results/`, `figures/`);
only `results/rev1/` is kept in the repository.

| Result | Make target | Underlying command | Output |
|---|---|---|---|
| Dataset and publication times | `make data` | `scripts/auxiliary/build_greek_2025_dataset.py`, `scripts/data/build_publication_sidecars.py`, `scripts/data/verify_dataset.py` | `data/processed/` (not redistributed) |
| Unit tests, Table "Settlement and model unit tests" (U1–U8) | `make test` | `scripts/make_tests_json.py` | `results/rev1/tests.json` |
| Settlement conformance (raw files read in the protocol's `market_clock`) | `make conformance` | `scripts/validation/validate_settlement.py` | `results/rev1/conformance.json`, `.csv` |
| Conference configuration, settlement v1 vs v2 (audit table; protocol-independent) | `make audit` | `scripts/run_audit.py` | `results/audit.json` |
| DAM forecaster error | — | `scripts/data/da_forecaster_error.py` | `results/rev1/da_forecaster_error.json` |
| Oracle bound and replay per day and seed | `make benchmark` (step 1) | `scripts/run_journal.py oracle` | `results/rev1/oracle_<task>.csv` |
| ORACLE-DELIVER bound `U_D` per day and seed (test split, `oracle.deliver_tasks`: T1, T2) | `make benchmark` (step 1b) | `scripts/run_journal.py oracle_deliver` | columns `U_D`, `oracle_D_*` of `results/rev1/oracle_<task>.csv` |
| Optimisation baselines per day and seed | `make benchmark` (step 2) | `scripts/run_journal.py baselines` | part of `results/rev1/benchmark_<task>.csv` |
| RL tuning, training, checkpoint selection, evaluation | `make rl` (part of `make benchmark`) | `scripts/rl/run_all.py tune`, `scripts/rl/run_all.py main`, `scripts/rl/collect_rl.py` | `results/rev1/rl_training/` (`tuning.json`, `hyperparameters.json`, `curves.csv`, `selections.csv`), part of `results/rev1/benchmark_<task>.csv` |
| Main table: anchors and scores per task; layer attribution; regime comparison | `make summary` (part of `make benchmark`) | `scripts/run_journal.py aggregate`, `scripts/summarise_journal.py` | `results/rev1/benchmark_<task>.csv`, `results/rev1/summary_<task>.json` |
| Paper tables A–E, H, I, RL details | — | `results/rev1/make_tables.py` | `results/rev1/tables_rev1.md` |
| Compute measurements | `make compute` | `scripts/measure_compute.py` | `results/rev1/compute.json` |
| Figure: daily scores on T1 | `make figures` | `figures/src/fig_scores_T1.py` | `figures/rev1/fig_scores_T1.pdf` |
| Figure: revenue decomposition on T1 | `make figures` | `figures/src/fig_decomposition_T1.py` | `figures/rev1/fig_decomposition_T1.pdf` |
| Figure: scores across tasks | `make figures` | `figures/src/fig_tasks.py` | `figures/rev1/fig_tasks.pdf` |
| Figure: RL validation curves and PPOL cost | `make figures` | `figures/src/fig_learning.py` | `figures/rev1/fig_learning.pdf` |
| Diagnostics of the rev1 brief, Part 0 (T1 per-step logs) | — | `results/rev1/part0/*.py` | `results/rev1/part0/` |

`make benchmark` runs all five tasks. T4-Ageing and T5-Combined read the
degradation cost `c_thr_eur_mwh: 5.7` from their task files. Override it only
for a sensitivity run, which writes to separate part directories:

```bash
make benchmark TASKS="T4-Ageing T5-Combined" C_THR=<EUR per MWh>
```

Intermediate per-job files are written to `results/rev1/parts/` and RL checkpoints
to `results/rev1/rl_training/runs/`; both are skipped when present, so an
interrupted run resumes. They are not version-controlled.

# Changelog

All notable changes are listed here. The project follows semantic versioning.

Every behavioural change has its own configuration flag (`SimOptions`,
`InformationConfig`, or a `make_unified_env` argument). `SimOptions.legacy()` /
`legacy_v04=True` sets all of them to the v0.4 conference behaviour;
`configs/conf_v1.yaml` is that configuration.

## 2.1.0 — unreleased (protocol rev1; tag `v2.1` prepared, not published)

Release notes: [RELEASE_NOTES_v2.1.md](RELEASE_NOTES_v2.1.md).

### Final technical revision (rev1 brief, Part A), protocol `configs/journal.yaml` rev1

`configs/journal.yaml` is now protocol **rev1** (results in `results/rev1/`,
figures in `figures/rev1/`). Protocol v3 is frozen as `configs/journal_v3.yaml`
(`PVBESS_PROTOCOL=configs/journal_v3.yaml`, or `make ... PROTOCOL=configs/journal_v3.yaml`)
and reproduces `results/v3/`. Evidence for every item: `results/status_rev1.md`.

| Flag (library default = v3; rev1 value) | New behaviour | v3 behaviour |
|---|---|---|
| `market_clock` (`"as_published"`; `"cet"`) | A3: the HEnEx results files and the IPTO files are in CET/CEST wall clock (on 2025-03-30 they skip 02:00, on 2025-10-26 they repeat 02:00). All market columns (DAM, intraday, imbalance, aFRR capacity and activation) and the per-auction sidecar prices are moved one hour later at load time (`pv_bess_gym/data/time_alignment.py`). IDA3 then covers 13:00–01:00 local. 2025-01-01 lacks its first market hour and is skipped (362 days; train 168). Publication times are not moved | The CET/CEST stamps were read as Athens time: every market value sat one hour before its delivery period |
| `radiation_stamp` (`"as_published"`; `"interval_centre"`) | A3: Open-Meteo hourly irradiance is the mean of the preceding hour (Open-Meteo documentation); GHI (actual, short-lead forecast, `_previous_day1`, `_previous_day2`) is moved 30 min earlier, to the centre of its interval. Temperature and wind (instantaneous) are not moved | Hourly means used as values at the end of their hour |
| `id_reference` (`"pv_only"`; `"d1_plan"`) | A1: the intraday rule and the intraday MPC expect to deliver the D-1 planned metered injection updated by the PV forecast change since D-1, `v = x + (1 − loss)(p̂PV_now − p̂PV_D-1)`. Rule: `y = mean v − mean (x + Y)` over the block, clipped to the position limits; MPC: `v` replaces `p̂PV` in the imbalance term | PV forecast only: the baselines sold when the D-1 plan charged the battery |
| `id_position_limit` (`"off"`; `"clip"`) | A2: the environment clips each intraday correction so that `−P_imp ≤ x_k + Y_k ≤ P_exp` on every step of its block; the intraday MPC carries the same bound | `x` and `Y` clipped separately to ±373 kW, so `x + Y` could reach 746 kW |

With `id_position_limit: "clip"` the oracles (ORACLE and ORACLE-DELIVER) get
the same bound `−P_imp ≤ x_t + Y_t ≤ P_exp` (read from the environment's
`SimOptions`), so `U` and `U_D` stay bounds of exactly the environment's
feasible set. Without it the oracle plan used positions up to 746 kW that the
environment then clips: on T1 test the replayed oracle profit fell from
`U_rep` = 1706.8 (v3) to 1174.7 EUR/day while `U` stayed at 1715.8, a valid
but much looser bound. Tests: `tests/test_rev1.py`.

Also: `oracle.deliver_tasks` (rev1: ORACLE-DELIVER and S_D for T1/T2 only, author decision, compute);
the oracle retries once with a 4x time limit when HiGHS returns no incumbent (column `retries`);
new paper figures in `figures/src/` (`make figures`) replace `scripts/plot_*.py`.

Validation script `scripts/validation/validate_settlement.py`: reads the raw
HEnEx/IPTO files in the wall clock given by the protocol's `market_clock`
(CET/CEST under rev1), and expands each HEnEx row over its own delivery
period. Before, a forward fill of up to 45 min could carry the last price of
the previous delivery day across midnight once the products became 15-minute
(from October 2025). Conformance: 0.0000 EUR largest daily deviation under both rev1 and v3.
`validate_settlement.py`, `da_forecaster_error.py`, `make test` and `make compute` now write to the
protocol's results folder (`results/rev1/`).

## 2.0.0 — journal version v3/v4 (not published)

### Realism fixes (agent brief v3), protocol `configs/journal_v3.yaml` (was `configs/journal.yaml`)

The version-4 results (`paper_inputs.md` v4) stay reproducible with
`PVBESS_PROTOCOL=configs/journal_v4.yaml` (all flags below at their version-4
value, realised DAM prices in T1/T3/T4). Version-3 results are written to
`results/v3/` and `figures/v3/` (`results_subdir` of the protocol).

| Flag (library default = version 4; v3 protocol value) | New behaviour | Version-4 behaviour |
|---|---|---|
| `da_weather` (`"short_lead"`; `"day_ahead"`) | R1: the D-1 layer (D-1 context observation, D-1 QP) sees Open-Meteo Previous Runs `_previous_day2`, issued >= 48 h before the valid time (`data/processed/weather_previous_runs_2025.csv`, `scripts/data/fetch_open_meteo_previous_runs.py`) | Short-lead Open-Meteo historical forecast for every layer |
| `id_weather` (`"same_as_da"`; `"latest"`) | R2: after the D-1 commitment, intraday/real-time observations show the short-lead forecast for delivery steps up to `short_lead_visible_h` (3 h, assumed publication delay) ahead and `_previous_day1` beyond; audited in test U7 (`observation_provenance()["pv_forecast_pub"]`) | Same series as D-1 |
| `pv_forecast_error_scale` | R5: kappa scales the residual of each series a layer receives | One series |
| task `da_price_info` (v3: `forecast` in all tasks) | R3: T1, T3, T4 use the naive `prev_day` DAM forecast before the D-1 commitment; `realised` remains an explicit optimistic variant | `realised` in T1, T3, T4 |
| `da_plant_consistent` (`False`; `True`) | R4: the D-1 QP of the baselines includes the 2 kW auxiliary load (supplied by the battery), self-discharge after the step, the 1 % metering loss, and a **balanced schedule** (DAM schedule = planned metered injection), so a plan delivered exactly gives zero imbalance | No auxiliary load, unmetered injection, deviations priced at the 160 EUR/MWh tracking penalty: whenever the DAM price exceeded it, the plan sold the full export capacity regardless of deliverability |
| oracle `deliver_epsilon_kwh` (v3: 0) | R6: ORACLE-DELIVER (`solve_oracle(allow_imbalance=False, deliver_epsilon_kwh=...)`, stage `run_journal.py oracle_deliver`), bound `U_D`, second score `S_D = (J - J_L)/(U_D - J_L)` next to `S` | One oracle |
| oracle `deliver_splits` (v3: `[test]`) | R6: ORACLE-DELIVER, `U_D` and `S_D` on the listed splits only; `aggregate` requires `U_D` there and leaves it empty elsewhere (compute; author decision D7) | — |

Also added: per-episode `scheduled_kwh`, `metered_kwh`,
`short_imbalance_steps`, `long_imbalance_steps` (`run_episode`) and the
`delivery` block in the summaries; `PVBESS_PROTOCOL` selects the protocol for
every script; tests `tests/test_realism_v3.py`.

### Behaviour changes and their flags

| Flag (default) | New behaviour | Previous behaviour |
|---|---|---|
| `settlement` (`"v2"`) | Scheduled energy includes requested aFRR activation, `E_sch = (q + a+ r+ - a- r-) dt`; activation settled once (received for up, paid back for down) | `"v1"`: activation in the imbalance volume and paid again as activation |
| `metering_losses` (`True`) | Grid loss factor applied to settled energy | Computed, not settled |
| `battery_aux_drains` (`True`) | Auxiliary load drains the battery | It charged the battery |
| `directional_taper` (`True`) | SOC power taper limits only the direction approaching the bound | Limited both directions |
| `energy_limited_power` (`True`) | **New in 2.0.** Realised power is limited by the energy between the SOC bounds; an empty battery does not serve the auxiliary load | The SOC clip could create energy: a discharge just inside the taper margin drew up to about 64 kWh that the battery did not hold |
| `degradation_on_ac_power` (`True`) | **New in 2.0.** Degradation cost `c_thr * abs(p_bat) * dt` | On DC-side power including the auxiliary load |
| `calendar_days` (`True`) | Episodes are calendar delivery days | Raw 96-row slices |
| `truncate_at_day_end` (`True`) | Last step returns `truncated=True` | `terminated=True` and a zero observation |
| `random_start` (`True`) | Seeded `reset()` samples the day | Always row 0 |
| `info_filter` (`False`; `True` in the benchmark tasks) | **New in 2.0.** Observations contain only values published at decision time: intraday entries show the newest auction already published (IDA1, IDA2, IDA3), imbalance entries show the observable DAM price | Realised intraday and imbalance prices of the whole day were observable |
| `da_price_info` (`"realised"`) | **New in 2.0.** `"forecast"`: the D-1 layer sees a naive DAM forecast (`da_forecaster`: `prev_day` or `prev_week`) until it commits | Realised DAM prices |
| `pv_forecast_error_scale` (1.0) | **Changed in 2.0.** Scales the weather-forecast residual *before* the PV model | Scaled the PV-power residual after the model |
| `c_thr_eur_mwh` (0) | **New in 2.0.** Degradation cost per MWh of throughput | `degradation_eur_per_kwh` (still accepted) |
| reset option `activation_seed` (none) | **New in 2.0.** A seeded aFRR activation draw per episode, identical for every controller and the oracle | Activation columns of the dataset |

### Added

- Benchmark tasks `T1-Nominal` ... `T5-Combined` (`configs/tasks/`, registered
  as `PVBESS-<task>-v2`) and the journal protocol `configs/journal.yaml`
  (split Jan-Jul / Aug-Sep / Oct-Dec, one day per episode, SOC 0.5, 10
  activation seeds). T4 and T5 need the author's `c_thr` and refuse to build
  without it.
- Baselines `da-only` and `da-opt-id-mpc` (recombinations of existing layers).
- `FixedD1Wrapper` and `FlatDispatchWrapper` (the former `DispatchRLWrapper`
  is their composition); constrained mode with a cost signal.
- PPO-Lagrangian through OmniSafe, unmodified (`pv_bess_gym/rl/omnisafe_env.py`).
- Auction prices and publication times read from the raw files
  (`scripts/data/build_publication_sidecars.py`).
- Tests U1-U8 of the journal brief, step-log determinism, oracle dominance,
  full-year observation audit; `make test` writes `results/tests.json`.
- `make conformance` with a prescribed schedule -> `results/conformance.json`.
- Run metadata (git commit, package and solver versions, seed, wall-clock) for
  results files.

### Changed

- Oracle reformulated to be an upper bound by construction: maximum
  efficiency, untapered limits, free energy disposal, free PV curtailment,
  shortfall and reserve-violation penalties priced exactly; solved per day
  and activation seed; replay profit available (`replay_oracle`).
- Dataset builder: the zonal clearing price is read from LOAD/SUPPLY rows
  only. Cross-border rows sometimes carry the neighbouring price; averaging
  them in shifted 912 DAM and 560 intraday steps by at most 0.005 EUR/MWh.
- Safety projection layer uses the battery's own power and energy limits.

## 1.0.0 — 2026-11-12

First benchmark release. **This is a breaking release**: the default
simulator and the dataset both change, so results are not comparable with
v0.4. The v0.4 behaviour is kept behind a flag (see *Reproducing v0.4*).

### Breaking changes

- **Settlement.** Activated aFRR energy is settled once, as balancing energy,
  and removed from the imbalance volume. In v0.4 upward activations were paid
  twice (as imbalance and as activation energy). Downward activation energy is
  now paid back by the plant instead of being credited.
- **Metering losses.** The grid loss factor is applied to the settled energy;
  v0.4 computed it but settled the unmetered value.
- **Battery.** The auxiliary load now drains the battery (it charged it in
  v0.4), and the power taper near an SOC bound only limits the direction that
  approaches the bound (in v0.4 an empty battery could not be recharged).
- **Episodes.** An episode is a calendar delivery day. The two DST-change days
  are excluded, so day indices after 30 March differ from v0.4. Episodes end
  with `truncated=True` and a regular final observation instead of
  `terminated=True` and zeros.
- **Seeding.** `reset(seed=...)` samples the delivery day; v0.4 always
  started at the first row. Pass `options={"day": k}` for a fixed day.
- **Defaults.** `MarketConfig` penalty defaults now match the factory and CLI
  (100 EUR/kW, 20 EUR/MWh); they were ten times larger.
- **Dataset.** The benchmark dataset is built from HEnEx and IPTO
  publications and is no longer shipped. The bundled `example_data/` is the
  v0.4 sample, whose DAM, intraday and aFRR capacity prices are proxies.

### Added

- Perfect-foresight MILP oracle with a proven revenue upper bound
  (`pv_bess_gym.oracle`).
- Benchmark protocol: chronological train/validation/test split, weekly
  evaluation blocks, three difficulty tiers in YAML, benchmark score
  (`pv_bess_gym.benchmark`, `configs/`).
- Difficulty knobs in the environment: PV forecast error scale, seeded price
  forecast noise, intraday gate closure, degradation cost
  (`InformationConfig`, `make_unified_env`).
- RL baselines on Stable-Baselines3 (PPO, SAC, PPO with a safety projection
  layer) with observation/action normalisation wrappers (`pv_bess_gym.rl`).
- Config-driven experiment runner with resumable, parallel stages and CSV/JSON
  results (`scripts/run_experiment.py`), and one script per paper table and
  figure (`scripts/paper/`).
- Settlement validation against the publishers' raw files
  (`scripts/validation/validate_settlement.py`).
- Dataset provenance checks (`scripts/data/verify_dataset.py`).
- Unit tests for every settlement leg, SOC limits, gate closure, seeding, the
  oracle and the RL wrappers; GitHub Actions CI.
- Gymnasium registration as `PVBESSGreece-v1`.

### Changed

- All exogenous signals, including PV power, are pre-computed: the
  environment runs at about 16,000 steps per second instead of about 7.
- The D-1 optimisation is compiled once per agent and re-solved with new data
  (about 50 times faster, same solution).
- Dataset builder: prices are held constant over their market time unit
  instead of being interpolated; real aFRR capacity prices from IPTO ISP1
  results; latest published version of imbalance prices; `.xls` downloads.

### Reproducing v0.4

```python
env = make_unified_env(signals_path="example_data",
                       plant_config_path="example_data/station_athens.json",
                       legacy_v04=True)
```

reproduces the v0.4 simulator on the v0.4 data to the cent (regression-tested
on the three DA-OPT controllers).

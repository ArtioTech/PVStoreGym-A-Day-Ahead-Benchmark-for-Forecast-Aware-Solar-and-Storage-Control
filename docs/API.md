# API reference

## Environment

```python
from pv_bess_gym import make_unified_env
env = make_unified_env(signals_path="data/processed",
                       plant_config_path="example_data/station_athens.json")
# or: gymnasium.make("PVBESSGreece-v1", signals_path=..., plant_config_path=...)
```

`make_unified_env` keyword arguments:

| Argument | Default | Meaning |
|---|---|---|
| `signals_path` | — | CSV/Excel file or directory of monthly `greek_YYYY_MM.csv` files |
| `plant_config_path` | — | plant JSON (`pv`, `battery`, `grid` sections) |
| `soc_init` | 0.10 | initial SOC when `reset` gets no `soc_init` option |
| `intraday_block_minutes` | 60 | interval between intraday decisions |
| `intraday_gate_closure_minutes` | 60 | lead time before an intraday trade is delivered |
| `forecast_horizon_steps` | 8 | look-ahead rows in `obs["forecast"]` |
| `violation_penalty` | 100 | EUR per kW of setpoint + reserve above the PCS rating |
| `shortfall_penalty` | 20 | EUR per MWh the battery could not deliver |
| `pv_forecast_error_scale` | 1.0 | 0 perfect, 1 archived forecast, 2 doubled errors |
| `price_noise_sigma` | 0.0 | price forecast noise, share of each price's std |
| `noise_seed` | 0 | seed of the forecast noise |
| `degradation_eur_per_kwh` | plant file | overrides the battery throughput cost |
| `episode_days` | all | days a seeded `reset()` samples from |
| `legacy_v04` | False | reproduce the v0.4 simulator |

### `reset(seed=None, options=None)`

Returns `(obs, info)`. With no options a day is sampled from `episode_days`
using the seed. Options:

| Key | Meaning |
|---|---|
| `day` | index of the delivery day (`env.num_days`, `env.day_date(k)`) |
| `start_idx` | raw row index of the first step (overrides `day`) |
| `soc_init` | initial SOC |
| `day_ahead_schedule_kw` | pre-commit a 96-step DAM schedule |
| `d1_cap_up_kw`, `d1_cap_down_kw` | pre-commit aFRR capacities |
| `id_adjustment_profile_kw` | pre-set intraday adjustments |

### `step(action)`

Returns `(obs, reward, terminated, truncated, info)`. `reward` is the settled
profit of the step in EUR. `terminated` is always False; `truncated` is True
on the last step of the delivery day.

Action dict (kW; out-of-range values are clipped):

| Key | Shape | Used when | Meaning |
|---|---|---|---|
| `day_ahead` | (96,) | `action_mask[0] == 1` | DAM schedule at the grid connection, export positive |
| `day_ahead_afrr` | (2,) | `action_mask[0] == 1` | `[cap_up, cap_down]` for the day |
| `intraday` | (1,) | `action_mask[1] == 1` | adjustment for the block after the gate closure |
| `real_time` | (1,) | every step | battery setpoint, discharge positive |

Observation dict:

| Key | Shape | Content |
|---|---|---|
| `physical` | (24,) | SOC, energy, PV actual and forecast, weather, prices, activation signals, commercial position, timers, time of day, committed capacities (index table in the README) |
| `forecast` | (H × 9,) | per look-ahead step: PV forecast, GHI and temperature forecast, DAM, intraday and imbalance price, aFRR capacity prices, scheduled position |
| `day_ahead_context` | (96 × 8,) | per step of the day: PV forecast, DAM, intraday and imbalance price, aFRR capacity prices, time-of-day sin/cos |
| `action_mask` | (3,) | `[day_ahead_due, intraday_due, 1]` |

Which price entries are forecasts rather than realised values is governed by
`price_noise_sigma` (see `InformationConfig`).

`info` holds the full settlement of the step: `dam_eur`, `id_eur`,
`imbalance_eur`, `afrr_capacity_eur`, `afrr_activation_eur`,
`degradation_eur`, `penalties_eur`, the realised prices, `metered_kw`,
`imbalance_kwh`, `soc`, `violation_kw`, `shortfall_mwh` and the timing flags.

## Settlement (v1)

Per 15-minute step, with `k = 0.25 h / 1000`:

| Leg | Formula |
|---|---|
| DAM | `k · Q_dam · P_dam` |
| Intraday | `k · Q_id · P_id` |
| Imbalance | `k · (metered − Q_dam − Q_id − activated) · P_imb` |
| aFRR capacity | `k · (cap_up · P_cap_up + cap_down · P_cap_down)` |
| aFRR energy | `k · (act_up · P_en_up − act_down · P_en_down)`; `P_en = P_imb` unless the dataset provides energy prices |
| Degradation | `throughput_kWh · cost` |
| Penalties | violation and shortfall penalties |

`metered = export · (1 − loss) − import · (1 + loss)`; `activated = act_up − act_down`
with `act = activation signal × committed capacity`.

## Controllers

```python
from pv_bess_gym import build_registered_algorithm, evaluate_agent
agent = build_registered_algorithm("da-opt-mpc", env)        # or da-opt-rbc, da-opt-rt-mpc
result = evaluate_agent(env, agent, day_indices=[300, 301, 302], soc_init=0.5)
result.total_profit, result.summary_df, result.combined_df
```

Any object with `reset()` and `act(obs) -> action` is a controller. All three
DA-OPT controllers share one D-1 optimisation, a convex quadratic program
(linear revenue and tracking terms, quadratic terminal-SOC term) solved with
CLARABEL. It has no integer variables.

## Oracle

```python
from pv_bess_gym.oracle import solve_oracle, OraclePlanAgent
res = solve_oracle(env, days=[300, 301, 302], soc_init=0.5)   # allow_imbalance=False for the reference
res.upper_bound_eur, res.objective_eur, res.mip_gap, res.breakdown, res.plan
evaluate_agent(env, OraclePlanAgent(env, res), day_indices=res.days, soc_init=0.5)
```

## Benchmark tasks

```python
from pv_bess_gym.benchmark import make_task_env, split_days, load_protocol
from pv_bess_gym.benchmark.episodes import run_episode
env = make_task_env("T1-Nominal")                     # protocol settings of configs/journal.yaml
agent = build_registered_algorithm("da-opt-mpc", env)
rows = [run_episode(env, agent, day, seed, soc_init=0.5)
        for day in split_days(env, "test") for seed in range(10)]
# or: gymnasium.make("PVBESS-T1-Nominal-v2")
```

`run_episode` returns profit, its components, delivery and constraint metrics
for one (day, activation seed). Scores need the oracle bound and the DA-ONLY
profit of the same episode (`scripts/run_journal.py aggregate`).

## RL

```python
from pv_bess_gym.rl import DispatchRLWrapper
from pv_bess_gym.rl.train import train_baseline, load_policy_agent
train_env = DispatchRLWrapper(env, projection=True)     # Box(-1, 1, (2,)) actions, flat observations
```

`train_baseline` saves `model.zip`, `vecnormalize.pkl` and `train.json`;
`load_policy_agent` turns a saved run into a controller for `evaluate_agent`.

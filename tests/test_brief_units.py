"""Tests U1-U8 of the journal brief, determinism, and oracle dominance.

Every expected value is computed here from closed-form expressions that do
not call the environment's settlement or battery code.
"""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from conftest import CAP_UP, DAM, IDP, IMB, action, make_env, make_signals
from pv_bess_gym import build_registered_algorithm
from pv_bess_gym.envs import InformationConfig, SimOptions
from pv_bess_gym.envs.information import InformationData
from pv_bess_gym.models.battery import BatteryConfig
from pv_bess_gym.oracle import replay_oracle, solve_oracle

DT = 0.25
K = DT / 1000.0  # kW over one step -> MWh
CFG = BatteryConfig()


def step_log_hash(log) -> str:
    rows = [{k: v for k, v in row.items()} for row in log]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=float).encode()).hexdigest()


def run(env, actions, **reset_options):
    env.reset(options={"day": 0, **reset_options})
    for act in actions:
        env.step(act)
    return list(env.log)


# --------------------------------------------------------------------- U1
def test_U1_dam_position_delivered_exactly():
    """Deliver the DAM position exactly: zero imbalance, reward = DAM revenue."""
    env = make_env()
    loss = env.plant.grid.loss_frac
    discharge = 40.0
    position = discharge * (1.0 - loss)              # what the meter records
    log = run(env, [action(env, dam=position, p_batt=discharge)] * 8)
    for row in log:
        # tolerance: actions are float32, so 39.6 kW is represented to about 4e-6 kW
        assert row["imbalance_kwh"] == pytest.approx(0.0, abs=1e-6)
        assert row["imbalance_eur"] == pytest.approx(0.0, abs=1e-7)
        assert row["reward"] == pytest.approx(K * position * DAM, abs=1e-6)


# --------------------------------------------------------------------- U2
@pytest.mark.parametrize("direction", ["up", "down"])
def test_U2_full_activation_settled_once_v2(direction):
    """Full activation delivered: settled once as activation, zero imbalance (v2)."""
    up = direction == "up"
    env = make_env(make_signals(act_up=1.0 if up else 0.0, act_down=0.0 if up else 1.0),
                   battery=BatteryConfig(aux_kw=0.0))
    loss = env.plant.grid.loss_frac
    cap = 100.0
    afrr = (cap, 0.0) if up else (0.0, cap)
    row = run(env, [action(env, afrr=afrr)])[0]
    sign = 1.0 if up else -1.0
    assert row["p_batt_real_kw"] == pytest.approx(sign * cap)
    assert row["afrr_activation_eur"] == pytest.approx(sign * K * cap * IMB)
    # only the metering loss on the delivered energy remains as imbalance
    assert row["imbalance_kwh"] == pytest.approx(-loss * cap * DT, abs=1e-9)


def test_U2_fails_under_v1_settlement():
    """Under the conference settlement the same case is paid twice."""
    env = make_env(make_signals(act_up=1.0), battery=BatteryConfig(aux_kw=0.0),
                   options=SimOptions(settlement="v1", metering_losses=False))
    row = run(env, [action(env, afrr=(100.0, 0.0))])[0]
    assert row["afrr_activation_eur"] == pytest.approx(K * 100.0 * IMB)
    assert row["imbalance_eur"] == pytest.approx(K * 100.0 * IMB)   # second payment: not zero
    assert row["imbalance_kwh"] != pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------- U3
@pytest.mark.parametrize("gate_minutes, first_open", [(60, 4), (120, 8)])
def test_U3_correction_goes_to_first_open_block(gate_minutes, first_open):
    """A correction decided at step 0 cannot touch closed steps; it lands on the first open block."""
    env = make_env(gate_minutes=gate_minutes)
    log = run(env, [action(env, intraday=30.0 if k == 0 else 0.0) for k in range(16)])
    changed = [k for k, row in enumerate(log) if row["id_adjustment_kw"] != 0.0]
    assert changed == list(range(first_open, first_open + 4))
    assert all(log[k]["id_eur"] == 0.0 for k in range(first_open))            # closed steps untouched
    assert log[first_open]["id_eur"] == pytest.approx(K * 30.0 * IDP)


# --------------------------------------------------------------------- U4
def test_U4_setpoint_beyond_pcs_is_clipped():
    env = make_env()
    row = run(env, [action(env, p_batt=10_000.0)])[0]
    assert row["p_batt_base_kw"] == pytest.approx(CFG.p_pcs_kw)


def test_U4_setpoint_beyond_soc_limit_is_projected_and_penalised():
    env = make_env(soc_init=CFG.soc_min)                 # empty battery
    row = run(env, [action(env, p_batt=200.0)], soc_init=CFG.soc_min)[0]
    assert row["p_batt_real_kw"] == pytest.approx(0.0, abs=1e-9)   # projected onto the feasible set
    expected_shortfall_mwh = 200.0 * DT / 1000.0
    assert row["shortfall_mwh"] == pytest.approx(expected_shortfall_mwh)
    assert row["penalties_eur"] == pytest.approx(env.market.fallback_shortfall_penalty_eur_per_mwh * expected_shortfall_mwh)


def test_U4_setpoint_plus_reserve_beyond_pcs_is_penalised():
    env = make_env()
    row = run(env, [action(env, afrr=(100.0, 0.0), p_batt=CFG.p_pcs_kw)])[0]
    assert row["violation_kw"] == pytest.approx(100.0)
    assert row["penalties_eur"] == pytest.approx(100.0 * env.market.violation_penalty_eur_per_kw, rel=1e-6)


# --------------------------------------------------------------------- U5
def test_U5_activation_above_available_energy_is_a_penalised_shortfall():
    env = make_env(make_signals(act_up=1.0), soc_init=CFG.soc_min, battery=BatteryConfig(aux_kw=0.0))
    row = run(env, [action(env, afrr=(200.0, 0.0))], soc_init=CFG.soc_min)[0]
    assert row["act_up_kw"] == pytest.approx(200.0)                 # requested
    assert row["p_batt_real_kw"] == pytest.approx(0.0, abs=1e-9)    # nothing to deliver
    shortfall_mwh = 200.0 * DT / 1000.0
    assert row["shortfall_mwh"] == pytest.approx(shortfall_mwh)
    assert row["penalties_eur"] == pytest.approx(env.market.fallback_shortfall_penalty_eur_per_mwh * shortfall_mwh)
    # the undelivered activation is also an imbalance against the requested trajectory
    assert row["imbalance_kwh"] == pytest.approx(-200.0 * DT)


# --------------------------------------------------------------------- U6
def _expected_energy(e0: float, p_real: float, cfg: BatteryConfig) -> float:
    """Unclipped SOC equation, written out independently of the battery class."""
    p_net = p_real + cfg.aux_kw
    eta = float(np.clip(cfg.eta_nom - cfg.eta_slope * min(1.0, abs(p_net) / cfg.p_pcs_kw), 0.85, 0.995))
    e1 = e0 - p_net * DT / eta if p_net >= 0 else e0 + (-p_net) * DT * eta
    return e1 * (1.0 - cfg.self_discharge_per_day * DT / 24.0)


def _energy_residuals(env, seed: int):
    cfg = env.plant.battery
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(options={"day": 0, "soc_init": float(rng.uniform(0.1, 0.9))})
    e_prev = float(obs["physical"][0]) * cfg.e_max_kwh
    e_prev = env.battery.e_kwh
    residuals = []
    for _ in range(env.episode_len):
        row = env.step(action(env, dam=float(rng.uniform(0, 300)), afrr=(40.0, 40.0),
                              intraday=float(rng.uniform(-100, 100)), p_batt=float(rng.uniform(-333, 333))))[4]
        e_now = row["soc"] * cfg.e_max_kwh
        residuals.append(e_now - _expected_energy(e_prev, row["p_batt_real_kw"], cfg))
        e_prev = e_now
    return np.array(residuals)


def test_U6_energy_balance_residual_on_random_admissible_days():
    """The realised SOC follows the SOC equation; only the unserved auxiliary load
    of an empty battery (at most aux*dt/eta per step) is not in the equation."""
    env = make_env(make_signals(sunny=True, act_up=0.3, act_down=0.3))
    cfg = env.plant.battery
    floor_allowance = cfg.aux_kw * DT / 0.85 + 1e-3
    for seed in range(5):
        residuals = _energy_residuals(env, seed)
        assert residuals.min() > -1e-6                    # energy is never lost unaccounted
        assert residuals.max() < floor_allowance          # and never created beyond the floor rule
        assert np.median(np.abs(residuals)) < 1e-6


def test_U6_detects_the_v10_energy_creation():
    """Without the energy limit the SOC clip created tens of kWh (regression guard)."""
    env = make_env(make_signals(sunny=True), options=SimOptions(energy_limited_power=False))
    worst = max(_energy_residuals(env, seed).max() for seed in range(5))
    assert worst > 10.0


# --------------------------------------------------------------------- U7
def _info_env(da_price_info="realised"):
    """Synthetic day with three auctions published at known times."""
    signals = make_signals(days=2)
    n = len(signals)
    signals["price_dam_eur_mwh"] = 100.0 + np.arange(n) % 96
    signals["price_imb_eur_mwh"] = 300.0 + np.arange(n) % 7
    ida1 = np.full(n, 110.0)
    ida2 = np.full(n, 120.0)
    ida3 = np.where((np.arange(n) % 96) >= 48, 130.0, np.nan)          # afternoon only
    signals["price_id_eur_mwh"] = np.where(np.isnan(ida3), ida2, ida3)  # final price used in settlement
    published = {}
    for day in pd.DatetimeIndex(signals["timestamp"]).normalize().unique():
        key = day.date().isoformat()
        published[(key, "DAM")] = day - pd.Timedelta(hours=9, minutes=40)
        published[(key, "IDA1")] = day - pd.Timedelta(hours=7, minutes=38)
        published[(key, "IDA2")] = day - pd.Timedelta(minutes=38)
        published[(key, "IDA3")] = day + pd.Timedelta(hours=11, minutes=22)
        published[(key, "ISP1")] = day - pd.Timedelta(hours=7)
    data = InformationData(ida1=ida1, ida2=ida2, ida3=ida3, published=published)
    env = make_env(signals, information=InformationConfig(info_filter=True, da_price_info=da_price_info), info_data=data)
    return env, signals


def audit_day(env, day: int, pub_lookup, realised_imb: np.ndarray) -> int:
    """Step through one day; return the number of observed values checked."""
    obs, _ = env.reset(options={"day": day})
    checked = 0
    for t in range(env.episode_len):
        prov = env.observation_provenance()
        ctx = obs["day_ahead_context"].reshape(env.episode_len, 8)
        for name, col in (("dam", 1), ("id", 2), ("imb", 3)):
            pub = np.array([pub_lookup(label) for label in prov[f"{name}_label"]])
            assert (pub <= t).all(), f"{name}: value observed at step {t} before publication"
            np.testing.assert_allclose(ctx[:, col], prov["values"][name].astype(np.float32), rtol=1e-6)
            checked += env.episode_len
        assert (prov["afrr_cap_pub"] <= t).all()
        # the realised imbalance price must not be in the observation
        assert not np.allclose(ctx[:, 3], realised_imb.astype(np.float32))
        obs = env.step(action(env))[0]
    return checked


def test_U7_observation_audit_synthetic():
    for mode in ("realised", "forecast"):
        env, signals = _info_env(mode)
        steps = {"dam": -39, "dam_forecast": -(10**9), "ida1": -31, "ida2": -3, "ida3": 46}
        imb = signals["price_imb_eur_mwh"].to_numpy()[96:192]
        assert audit_day(env, 1, steps.__getitem__, imb) == 96 * 96 * 3

        obs, _ = env.reset(options={"day": 1})
        ctx = obs["day_ahead_context"].reshape(96, 8)
        assert set(np.unique(ctx[:, 2])) == {120.0}                    # IDA2 everywhere at step 0
        for _ in range(46):
            obs = env.step(action(env))[0]
        ctx = obs["day_ahead_context"].reshape(96, 8)
        assert set(np.unique(ctx[:48, 2])) == {120.0} and set(np.unique(ctx[48:, 2])) == {130.0}


def test_U7_forecast_mode_hides_realised_dam_until_commitment():
    env, signals = _info_env("forecast")
    obs, _ = env.reset(options={"day": 1})
    truth = signals["price_dam_eur_mwh"].to_numpy()[96:192].astype(np.float32)
    previous_day = signals["price_dam_eur_mwh"].to_numpy()[0:96].astype(np.float32)
    np.testing.assert_allclose(obs["day_ahead_context"].reshape(96, 8)[:, 1], previous_day)
    obs = env.step(action(env))[0]
    np.testing.assert_allclose(obs["day_ahead_context"].reshape(96, 8)[:, 1], truth)


# --------------------------------------------------------------------- U8
def test_U8_inactive_action_keys_are_ignored():
    env = make_env(make_signals(sunny=True, act_up=0.2))
    rng = np.random.default_rng(0)
    first = action(env, dam=80.0, afrr=(50.0, 30.0), intraday=20.0, p_batt=10.0)
    clean, noisy = [first], [first]
    for k in range(1, 24):
        intraday = 15.0 if k % 4 == 0 else 0.0
        clean.append(action(env, intraday=intraday, p_batt=10.0))
        garbage = action(env, dam=float(rng.uniform(-300, 300)), afrr=tuple(rng.uniform(0, 300, 2)),
                         intraday=intraday if k % 4 == 0 else float(rng.uniform(-300, 300)), p_batt=10.0)
        noisy.append(garbage)
    assert step_log_hash(run(env, clean)) == step_log_hash(run(env, noisy))


# ------------------------------------------------------------- determinism
def _fixed_actions(env, reserve):
    rng = np.random.default_rng(42)
    return [action(env, dam=60.0, afrr=reserve, intraday=float(rng.uniform(-50, 50)), p_batt=float(rng.uniform(-150, 150)))
            for _ in range(env.episode_len)]


def test_determinism_same_seed_same_actions_same_step_log_hash():
    hashes = set()
    for _ in range(2):
        env = make_env(make_signals(sunny=True))
        hashes.add(step_log_hash(run(env, _fixed_actions(env, (60.0, 40.0)), activation_seed=3)))
    assert len(hashes) == 1


def test_different_seeds_differ_only_in_stochastic_signals():
    logs = {}
    for seed in (1, 2):
        env = make_env(make_signals(sunny=True))
        logs[seed] = run(env, _fixed_actions(env, (60.0, 40.0)), activation_seed=seed)
        logs[seed, "ratios"] = (env._ep_act_up.copy(), env._ep_act_dn.copy())
    assert not np.array_equal(logs[1, "ratios"][0], logs[2, "ratios"][0])        # the stochastic signal differs
    exogenous = ["timestamp", "pv_kw", "pv_fcst_kw", "price_dam_eur_mwh", "price_id_eur_mwh", "price_imb_eur_mwh",
                 "dam_position_kw", "id_adjustment_kw", "p_batt_base_kw", "cap_up_commit_kw", "cap_down_commit_kw",
                 "dam_eur", "id_eur", "afrr_capacity_eur"]
    for a, b in zip(logs[1], logs[2]):
        assert all(a[key] == b[key] for key in exogenous)
    # without committed reserve the activation draw has no effect at all
    hashes = set()
    for seed in (1, 2):
        env = make_env(make_signals(sunny=True))
        hashes.add(step_log_hash(run(env, _fixed_actions(env, (0.0, 0.0)), activation_seed=seed)))
    assert len(hashes) == 1


# ----------------------------------------------------------- oracle >= all
def test_oracle_bounds_every_baseline_on_every_day_and_seed():
    signals = make_signals(days=2, sunny=True)
    n = np.arange(len(signals))
    signals["price_dam_eur_mwh"] = 100.0 + 60.0 * np.sin(2 * np.pi * n / 96.0)
    signals["price_id_eur_mwh"] = signals["price_dam_eur_mwh"] + 8.0 * np.cos(2 * np.pi * n / 48.0)
    signals["price_imb_eur_mwh"] = signals["price_dam_eur_mwh"] + 25.0 * np.sin(2 * np.pi * n / 17.0)
    env = make_env(signals)
    for day in (0, 1):
        for seed in (0, 1):
            oracle = solve_oracle(env, day, soc_init=0.5, activation_seed=seed, time_limit_s=60.0)
            replay = replay_oracle(env, oracle, soc_init=0.5)
            assert replay <= oracle.upper_bound_eur + 1e-6
            for name in ["da-only", "da-opt-rbc", "da-opt-rt-mpc", "da-opt-id-mpc", "da-opt-mpc"]:
                agent = build_registered_algorithm(name, env)
                agent.reset()
                obs, _ = env.reset(options={"day": day, "soc_init": 0.5, "activation_seed": seed})
                profit, done = 0.0, False
                while not done:
                    obs, reward, term, trunc, _ = env.step(agent.act(obs))
                    profit += reward
                    done = term or trunc
                assert profit <= oracle.upper_bound_eur + 1e-6, (name, day, seed)
            # random policies, including ones that violate limits and miss delivery
            env.action_space.seed(seed)
            env.reset(options={"day": day, "soc_init": 0.5, "activation_seed": seed})
            profit = sum(env.step(env.action_space.sample())[1] for _ in range(env.episode_len))
            assert profit <= oracle.upper_bound_eur + 1e-6

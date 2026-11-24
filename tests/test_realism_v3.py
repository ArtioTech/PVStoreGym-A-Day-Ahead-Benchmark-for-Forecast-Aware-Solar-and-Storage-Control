"""Tests of the realism fixes R1-R6 (agent brief v3). Need the built dataset."""

from pathlib import Path

import numpy as np
import pytest

from pv_bess_gym.agents.registry import build_registered_algorithm
from pv_bess_gym.benchmark.episodes import run_episode
from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, split_days
from pv_bess_gym.oracle import solve_oracle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(
    not (DATA / "weather_previous_runs_2025.csv").exists() or not (DATA / "publication_times_2025.csv").exists(),
    reason="built dataset with the Open-Meteo Previous Runs file not present",
)
R4_TOL_KWH = 1e-4      # float32 actions; observed maximum 5e-6 kWh


def test_R4_balanced_d1_plan_delivered_exactly_has_zero_imbalance():
    """Perfect forecast, no activation: executing the D-1 plan meters the schedule at every step."""
    env = make_task_env("T1-Nominal", pv_forecast_error_scale=0.0, da_price_info="realised", da_plant_consistent=True)
    agent = build_registered_algorithm("da-only", env, {})
    days = split_days(env, "test")[::9]
    for day in days:
        agent.reset()
        obs, _ = env.reset(options={"day": day, "soc_init": 0.5, "activation_seed": 0})
        plan = agent._solve_d1_cooptimization(obs)
        assert plan is not None
        assert np.max(plan["planned_imbalance_abs_kw"]) < 1e-6
        for t in range(env.episode_len):
            action = {"day_ahead": plan["schedule"].astype(np.float32), "day_ahead_afrr": np.zeros(2, np.float32),
                      "intraday": np.zeros(1, np.float32), "real_time": np.array([plan["battery_kw"][t]], np.float32)}
            obs, _, _, _, info = env.step(action)
            assert abs(info["imbalance_kwh"]) < R4_TOL_KWH, (day, t, info["imbalance_kwh"])


def test_R4_flag_off_reproduces_version_4_plan_imbalance():
    """Without the flag the D-1 plan may carry planned imbalance (version-4 behaviour)."""
    env = make_task_env("T1-Nominal", pv_forecast_error_scale=0.0, da_price_info="realised", da_plant_consistent=False)
    agent = build_registered_algorithm("da-only", env, {})
    obs, _ = env.reset(options={"day": split_days(env, "test")[0], "soc_init": 0.5, "activation_seed": 0})
    assert np.max(agent._solve_d1_cooptimization(obs)["planned_imbalance_abs_kw"]) > 1.0


@pytest.mark.parametrize("name", ["T1-Nominal", "T2-Forecast"])
def test_U7_weather_full_year_no_forecast_before_its_issue_time(name):
    """R1/R2 audit: every PV forecast in the observation was issued before decision time."""
    env = make_task_env(name)
    assert env.information.da_weather == "day_ahead" and env.information.id_weather == "latest"
    n = env.episode_len
    visible = env._wx_visible_steps
    checked = 0
    for day in range(env.num_days):
        env.reset(options={"day": day, "soc_init": 0.5, "activation_seed": 0})
        pre = env._observed_weather(0, False)
        assert set(pre["label"]) == {"weather_d2"}
        assert np.all(pre["pub"] <= -n)                    # issued no later than 00:00 of D-1
        assert np.allclose(pre["pv"], env._ep_wx["d2"][0])
        for t in range(n):
            seen = env._observed_weather(t, True)
            assert np.all(seen["pub"] <= t), (day, t)
            v = np.arange(n)
            near = v <= t + visible
            assert set(seen["label"][near]) <= {"weather_short_lead"}
            assert set(seen["label"][~near]) <= {"weather_d1"}
            assert np.allclose(seen["pv"][near], env._ep_wx["short_lead"][0][near])
            assert np.allclose(seen["pv"][~near], env._ep_wx["d1"][0][~near])
            checked += 1
    assert checked == env.num_days * n


def test_U7_weather_provenance_is_exposed_in_the_observation_audit():
    env = make_task_env("T1-Nominal")
    agent = build_registered_algorithm("da-only", env, {})
    obs, _ = env.reset(options={"day": 10, "soc_init": 0.5, "activation_seed": 0})
    prov = env.observation_provenance()
    assert set(prov["pv_forecast_label"]) == {"weather_d2"}
    for _ in range(40):
        obs, *_ = env.step(agent.act(obs))
    prov = env.observation_provenance()
    assert np.all(prov["pv_forecast_pub"] <= prov["t"])
    # what the agent sees as PV forecast is the audited series
    assert np.allclose(obs["day_ahead_context"].reshape(env.episode_len, -1)[:, 0],
                       env._observed_weather(prov["t"], True)["pv"], atol=1e-3)


def test_R5_kappa_scales_each_layer_residual():
    e1 = make_task_env("T2-Forecast", pv_forecast_error_scale=1.0)
    e2 = make_task_env("T2-Forecast")      # kappa = 2
    for key in ("short_lead", "d1", "d2"):
        g1, g2 = e1._wx[key][1], e2._wx[key][1]
        actual = e1._ghi
        unclipped = (g2 > 0)
        assert np.allclose((g2 - actual)[unclipped], 2.0 * (g1 - actual)[unclipped], atol=1e-6)


@pytest.mark.parametrize("day_offset", [13, 78])
def test_R6_oracle_deliver_between_da_only_and_oracle(day_offset):
    """ORACLE-DELIVER <= ORACLE and ORACLE-DELIVER >= DA-ONLY on every activation seed of a test day."""
    proto = load_protocol()
    env = make_task_env("T1-Nominal")
    day = split_days(env, "test")[day_offset]
    agent = build_registered_algorithm("da-only", env, {})
    for seed in (0, 5):
        full = solve_oracle(env, day, soc_init=0.5, activation_seed=seed, time_limit_s=120)
        deliver = solve_oracle(env, day, soc_init=0.5, activation_seed=seed, allow_imbalance=False,
                               deliver_epsilon_kwh=float(proto["oracle"]["deliver_epsilon_kwh"]), time_limit_s=120)
        j_l = run_episode(env, agent, day, seed, 0.5)["profit_eur"]
        assert deliver.objective_eur <= full.upper_bound_eur + 1e-6
        assert deliver.upper_bound_eur >= j_l - 1e-6

"""Tests of the rev1 corrections (A1 intraday reference, A2 position limit, A3 time stamps). Need the built dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pv_bess_gym.agents.registry import build_registered_algorithm
from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, split_days

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RAW_HENEX = ROOT / "data" / "raw" / "henex" / "2025_EL-DAM-IDAs_Results" / "Results"
pytestmark = pytest.mark.skipif(
    not (DATA / "weather_previous_runs_2025.csv").exists() or not (DATA / "publication_times_2025.csv").exists(),
    reason="built dataset with the sidecar files not present",
)
V3 = load_protocol(ROOT / "configs" / "journal_v3.yaml")
REV1 = load_protocol(ROOT / "configs" / "journal.yaml")


@pytest.fixture(scope="module")
def envs():
    return (make_task_env("T1-Nominal", protocol=V3), make_task_env("T1-Nominal", protocol=REV1))


def test_A3_raw_henex_files_are_cet_wall_clock():
    """Evidence for the correction: the HEnEx DAM file of 2025-03-30 skips 02:00 (CET), not 03:00 (Athens)."""
    files = sorted(RAW_HENEX.rglob("20250330_EL-DAM_Results_EN_v*.xlsx"))
    if not files:
        pytest.skip("raw HEnEx files not present")
    df = pd.read_excel(files[-1])
    hours = sorted(pd.to_datetime(df["DELIVERY_MTU"]).dt.hour.unique())
    assert 2 not in hours and 3 in hours


def test_A3_market_moved_one_hour_later_weather_unchanged(envs):
    v3, rev1 = envs
    for a, b in [(rev1._dam, v3._dam), (rev1._idp, v3._idp), (rev1._imb, v3._imb), (rev1._cup, v3._cup),
                 (rev1._cdn, v3._cdn), (rev1.info_data.ida3, v3.info_data.ida3)]:
        assert np.allclose(a[4:], b[:-4], equal_nan=True)
        assert np.isnan(a[:4]).all()
    assert np.array_equal(rev1._temp, v3._temp)          # instantaneous variables are not moved
    assert np.array_equal(rev1._wind, v3._wind)


def test_A3_ida3_starts_at_13_local_and_publication_times_unchanged(envs):
    v3, rev1 = envs
    day = split_days(rev1, "test", REV1)[10]
    s = rev1.day_start_index(day)
    hours = rev1._ts[s:s + 96].hour
    have = ~np.isnan(rev1.info_data.ida3[s:s + 96])
    assert not have[(hours >= 1) & (hours < 13)].any()
    assert have[hours >= 13].all()
    assert v3.info_data.published == rev1.info_data.published


def test_A3_first_day_without_market_data_is_skipped(envs):
    v3, rev1 = envs
    assert v3.day_date(0) == "2025-01-01" and rev1.day_date(0) == "2025-01-02"
    assert rev1.num_days == v3.num_days - 1
    assert len(split_days(rev1, "test", REV1)) == len(split_days(v3, "test", V3))


def test_A3_irradiance_moved_30_min_earlier(envs):
    v3, rev1 = envs
    assert np.allclose(rev1._ghi[:-2], v3._ghi[2:])
    assert np.allclose(rev1._ghi_f[:-2], v3._ghi_f[2:])
    for key in ("ghi_d1", "ghi_d2"):
        assert np.allclose(rev1.info_data.weather_lead[key][:-2], v3.info_data.weather_lead[key][2:])


def test_A3_flags_off_reproduce_v3_arrays():
    a = make_task_env("T1-Nominal", protocol=V3)
    b = make_task_env("T1-Nominal", protocol=V3, market_clock="as_published", radiation_stamp="as_published")
    assert np.array_equal(a._dam, b._dam) and np.array_equal(a._ghi, b._ghi) and a.num_days == 363


def test_A2_position_clip_bounds_x_plus_y():
    env = make_task_env("T1-Nominal", protocol=REV1)
    env.reset(options={"day": split_days(env, "test", REV1)[0], "soc_init": 0.5, "activation_seed": 0})
    x = np.full(env.episode_len, 300.0, dtype=np.float32)
    act = {"day_ahead": x, "day_ahead_afrr": np.zeros(2, np.float32),
           "intraday": np.array([300.0], np.float32), "real_time": np.zeros(1, np.float32)}
    _, _, _, _, info = env.step(act)
    assert info["intraday_target_start"] == env.intraday_gate_steps
    assert env.id_adjustment_profile_kw[env.intraday_gate_steps] == pytest.approx(73.0)
    env_off = make_task_env("T1-Nominal", protocol=V3)
    env_off.reset(options={"day": split_days(env_off, "test", V3)[0], "soc_init": 0.5, "activation_seed": 0})
    env_off.step(act)
    assert env_off.id_adjustment_profile_kw[env_off.intraday_gate_steps] == pytest.approx(300.0)


@pytest.mark.parametrize("controller", ["da-opt-rbc", "da-opt-id-mpc"])
def test_A1_d1_plan_reference_trades_nothing_without_forecast_change(controller):
    """With a perfect PV forecast the D-1 and intraday forecasts agree, so the expected
    delivery equals the D-1 schedule; the rule trades nothing and the MPC has no imbalance reason to trade."""
    env = make_task_env("T1-Nominal", protocol=REV1, pv_forecast_error_scale=0.0)
    agent = build_registered_algorithm(controller, env)
    agent.reset()
    obs, _ = env.reset(options={"day": split_days(env, "test", REV1)[5], "soc_init": 0.5, "activation_seed": 0})
    done, checked = False, 0
    while not done:
        if obs["action_mask"][1] > 0.5 and agent._day_ahead_schedule is not None:
            target = agent._target_slice()
            if target.start < target.stop:
                v = agent._expected_delivery_kw(target, obs)
                assert np.allclose(v, agent._day_ahead_schedule[target][: len(v)], atol=1e-3)
                checked += 1
        obs, _, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
        if controller == "da-opt-rbc":
            assert abs(info["id_adjustment_kw"]) < 1e-3
    assert checked > 0


def test_A1_pv_only_flag_keeps_v3_rule():
    env = make_task_env("T1-Nominal", protocol=V3)
    agent = build_registered_algorithm("da-opt-rbc", env)
    assert agent.id_reference == "pv_only" and not agent._position_clip


def test_A2_oracle_respects_position_limit():
    """With the clip flag the oracle plan keeps -P_imp <= x + Y <= P_exp."""
    from pv_bess_gym.oracle import solve_oracle

    env = make_task_env("T1-Nominal", protocol=REV1)
    day = split_days(env, "test", REV1)[0]
    res = solve_oracle(env, day, soc_init=0.5, activation_seed=0, time_limit_s=60.0)
    q = res.plan["day_ahead_kw"][0] + res.plan["id_adjustment_kw"][0]
    assert q.max() <= env.plant.grid.export_cap_kw + 1e-6
    assert q.min() >= -env.plant.grid.import_cap_kw - 1e-6

"""Exact cash-flow tests for each settlement leg (zero-PV, flat-price market)."""

import numpy as np
import pytest

from conftest import CAP_DN, CAP_UP, DAM, IDP, IMB, action, make_env, make_signals
from pv_bess_gym.models.grid import GridConfig

DT = 0.25


def eur(kw, price):
    return kw * DT / 1000.0 * price


def run_day(env, **kw):
    env.reset(options={"day": 0, **kw.pop("reset", {})})
    infos = []
    for _ in range(env.episode_len):
        infos.append(env.step(action(env, **kw))[4])
    return infos


def test_dam_and_imbalance_legs():
    """Sell 50 kW day-ahead, deliver nothing: paid at DAM, bought back at imbalance."""
    env = make_env()
    info = run_day(env, dam=50.0)[0]
    assert info["dam_eur"] == pytest.approx(eur(50.0, DAM))
    assert info["imbalance_eur"] == pytest.approx(eur(-50.0, IMB))
    assert info["id_eur"] == 0.0
    assert info["reward"] == pytest.approx(eur(50.0, DAM - IMB))


def test_reward_is_sum_of_components():
    env = make_env(make_signals(sunny=True, act_up=0.3, act_down=0.1))
    env.reset(seed=3)
    env.action_space.seed(3)
    for _ in range(env.episode_len):
        _, reward, _, _, i = env.step(env.action_space.sample())
        total = (
            i["dam_eur"] + i["id_eur"] + i["imbalance_eur"] + i["afrr_capacity_eur"]
            + i["afrr_activation_eur"] - i["degradation_eur"] - i["penalties_eur"]
        )
        assert reward == pytest.approx(total, abs=1e-9)


def test_delivered_energy_settles_with_metering_losses():
    """Discharge 40 kW against a 40 kW schedule: the 1% metering loss is an imbalance."""
    env = make_env()
    info = run_day(env, dam=40.0, p_batt=40.0)[0]
    assert info["metered_kw"] == pytest.approx(40.0 * 0.99)
    assert info["imbalance_eur"] == pytest.approx(eur(40.0 * 0.99 - 40.0, IMB))


def test_no_losses_means_no_imbalance_when_schedule_is_met():
    env = make_env(grid=GridConfig(loss_frac=0.0))
    info = run_day(env, dam=40.0, p_batt=40.0)[0]
    assert info["imbalance_eur"] == pytest.approx(0.0, abs=1e-12)
    assert info["reward"] == pytest.approx(eur(40.0, DAM))


@pytest.mark.parametrize("gate_minutes, first_step", [(0, 0), (60, 4), (120, 8)])
def test_intraday_gate_closure_delay(gate_minutes, first_step):
    """An intraday trade placed at step t only reaches delivery after the gate closure."""
    env = make_env(gate_minutes=gate_minutes)
    env.reset(options={"day": 0})
    infos = [env.step(action(env, intraday=30.0 if k == 0 else 0.0))[4] for k in range(16)]
    adjusted = [k for k, i in enumerate(infos) if i["id_adjustment_kw"] != 0.0]
    assert adjusted == list(range(first_step, first_step + 4))
    assert infos[first_step]["id_eur"] == pytest.approx(eur(30.0, IDP))
    assert infos[0]["intraday_applied"] and not infos[1]["intraday_applied"]


def test_intraday_decisions_only_on_block_boundaries(env):
    obs, _ = env.reset(options={"day": 0})
    due = []
    for _ in range(12):
        due.append(bool(obs["action_mask"][1]))
        obs = env.step(action(env))[0]
    assert due == [True, False, False, False] * 3


def test_intraday_cannot_reach_past_the_delivery_day(env):
    env.reset(options={"day": 0})
    for _ in range(env.episode_len - 4):
        env.step(action(env))
    info = env.step(action(env, intraday=25.0))[4]
    assert info["intraday_target_start"] == info["intraday_target_stop"] == env.episode_len


def test_afrr_capacity_payment():
    env = make_env()
    infos = run_day(env, afrr=(100.0, 60.0))
    expected = (100.0 / 1000.0 * CAP_UP + 60.0 / 1000.0 * CAP_DN) * DT
    assert all(i["afrr_capacity_eur"] == pytest.approx(expected) for i in infos)
    assert all(i["cap_up_commit_kw"] == 100.0 and i["cap_down_commit_kw"] == 60.0 for i in infos)


def test_afrr_capacity_is_fixed_after_d1():
    env = make_env()
    env.reset(options={"day": 0})
    env.step(action(env, afrr=(100.0, 60.0)))
    info = env.step(action(env, afrr=(0.0, 0.0)))[4]
    assert info["cap_up_commit_kw"] == 100.0


def test_afrr_up_activation_is_paid_once():
    """v1: activated energy is settled as balancing energy only, not also as imbalance."""
    env = make_env(make_signals(act_up=0.5), grid=GridConfig(loss_frac=0.0))
    info = run_day(env, afrr=(100.0, 0.0))[0]
    assert info["act_up_kw"] == pytest.approx(50.0)
    assert info["p_batt_real_kw"] == pytest.approx(50.0)
    assert info["afrr_activation_eur"] == pytest.approx(eur(50.0, IMB))
    assert info["imbalance_eur"] == pytest.approx(0.0, abs=1e-12)


def test_afrr_down_activation_is_paid_back():
    env = make_env(make_signals(act_down=0.5), grid=GridConfig(loss_frac=0.0))
    info = run_day(env, afrr=(0.0, 100.0))[0]
    assert info["p_batt_real_kw"] == pytest.approx(-50.0)
    assert info["afrr_activation_eur"] == pytest.approx(-eur(50.0, IMB))
    assert info["imbalance_eur"] == pytest.approx(0.0, abs=1e-12)


def test_legacy_double_counts_up_activation():
    env = make_env(make_signals(act_up=0.5), legacy=True)
    env.reset()
    info = env.step(action(env, afrr=(100.0, 0.0)))[4]
    assert info["afrr_activation_eur"] == pytest.approx(eur(50.0, IMB))
    assert info["imbalance_eur"] == pytest.approx(eur(50.0, IMB))


def test_reserve_violation_penalty():
    """Setpoint plus reserved capacity above the PCS rating is penalised per kW."""
    env = make_env()
    ppcs = env.plant.battery.p_pcs_kw
    info = run_day(env, afrr=(100.0, 0.0), p_batt=ppcs)[0]
    assert info["violation_kw"] == pytest.approx(100.0)
    assert info["penalties_eur"] >= 100.0 * env.market.violation_penalty_eur_per_kw


def test_degradation_cost():
    from pv_bess_gym.models.battery import BatteryConfig

    env = make_env(battery=BatteryConfig(deg_eur_per_kwh_throughput=0.02, aux_kw=0.0))
    info = run_day(env, p_batt=100.0)[0]
    assert info["degradation_eur"] == pytest.approx(100.0 * DT * 0.02)

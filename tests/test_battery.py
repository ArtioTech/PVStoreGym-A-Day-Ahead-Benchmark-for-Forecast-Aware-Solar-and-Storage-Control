import numpy as np
import pytest

from conftest import action, make_env, make_signals
from pv_bess_gym.models.battery import Battery, BatteryConfig


def test_soc_stays_within_limits_under_random_actions():
    env = make_env(make_signals(sunny=True, act_up=0.4, act_down=0.4))
    cfg = env.plant.battery
    for seed in range(5):
        env.reset(seed=seed)
        env.action_space.seed(seed)
        for _ in range(env.episode_len):
            info = env.step(env.action_space.sample())[4]
            assert cfg.soc_min - 1e-9 <= info["soc"] <= cfg.soc_max + 1e-9


@pytest.mark.parametrize("p_batt, bound", [(373.0, "soc_min"), (-373.0, "soc_max")])
def test_full_power_saturates_at_the_bound(p_batt, bound):
    env = make_env()
    env.reset(options={"day": 0})
    for _ in range(env.episode_len):
        info = env.step(action(env, p_batt=p_batt))[4]
    assert info["soc"] == pytest.approx(getattr(env.plant.battery, bound), abs=5e-3)
    assert abs(info["p_batt_real_kw"]) < 0.2 * abs(p_batt)
    assert info["shortfall_mwh"] > 0.0


def test_auxiliary_load_drains_an_idle_battery():
    cfg = BatteryConfig(self_discharge_per_day=0.0)
    battery = Battery(cfg, soc_init=0.5)
    e0 = battery.e_kwh
    battery.step(0.0, dt_h=0.25)
    assert battery.e_kwh == pytest.approx(e0 - cfg.aux_kw * 0.25 / battery.eta(cfg.aux_kw))


def test_legacy_auxiliary_load_charged_the_battery():
    cfg = BatteryConfig(self_discharge_per_day=0.0)
    battery = Battery(cfg, soc_init=0.5, legacy_v04=True)
    e0 = battery.e_kwh
    battery.step(0.0, dt_h=0.25)
    assert battery.e_kwh > e0


def test_empty_battery_can_be_charged_again():
    cfg = BatteryConfig()
    battery = Battery(cfg, soc_init=cfg.soc_min)
    assert battery.step(-100.0, dt_h=0.25)["p_real_kw"] == pytest.approx(-100.0)
    assert battery.step(373.0, dt_h=0.25)["p_real_kw"] < 373.0  # discharge still tapered

    legacy = Battery(cfg, soc_init=cfg.soc_min, legacy_v04=True)
    assert legacy.step(-100.0, dt_h=0.25)["p_real_kw"] == 0.0


def test_round_trip_loses_energy():
    cfg = BatteryConfig(aux_kw=0.0, self_discharge_per_day=0.0)
    battery = Battery(cfg, soc_init=0.5)
    e0 = battery.e_kwh
    battery.step(-200.0, dt_h=0.25)
    battery.step(200.0, dt_h=0.25)
    assert battery.e_kwh < e0


def test_vectorised_pv_matches_scalar_model():
    env = make_env(make_signals(sunny=True))
    df = env.df_all
    for i in np.linspace(0, len(df) - 1, 25).astype(int):
        scalar = env.pv_model.ac_power_kw(df.timestamp[i], df.ghi_wm2[i], df.temp_air_c[i], df.wind_mps[i])
        assert env._pv_act[i] == pytest.approx(scalar, abs=1e-9)

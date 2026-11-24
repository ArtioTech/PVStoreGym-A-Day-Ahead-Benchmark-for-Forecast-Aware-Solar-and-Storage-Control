import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from conftest import action, make_env, make_signals
from pv_bess_gym.envs import InformationConfig


def test_check_env_v1():
    check_env(make_env(make_signals(sunny=True)), skip_render_check=True)


def test_check_env_legacy():
    check_env(make_env(make_signals(sunny=True), legacy=True), skip_render_check=True)


def test_reset_seed_selects_day_deterministically(env):
    days = [env.reset(seed=s)[1]["day"] for s in range(40)]
    again = [env.reset(seed=s)[1]["day"] for s in range(40)]
    assert days == again
    assert len(set(days)) > 1, "different seeds must reach different days"


def test_same_seed_same_trajectory():
    def rollout(seed):
        e = make_env(make_signals(sunny=True, act_up=0.2))
        obs, _ = e.reset(seed=seed)
        e.action_space.seed(seed)
        rewards = []
        for _ in range(e.episode_len):
            obs, r, term, trunc, _ = e.step(e.action_space.sample())
            rewards.append(r)
        return np.array(rewards), obs["physical"]

    r1, o1 = rollout(7)
    r2, o2 = rollout(7)
    np.testing.assert_array_equal(r1, r2)
    np.testing.assert_array_equal(o1, o2)


def test_episode_days_restrict_sampling():
    e = make_env()
    e.set_episode_days([2])
    assert {e.reset(seed=s)[1]["day"] for s in range(10)} == {2}


def test_explicit_day_and_start_idx(env):
    _, info = env.reset(options={"day": 3})
    assert info["start_idx"] == 3 * 96
    _, info = env.reset(options={"start_idx": 96})
    assert info["start_idx"] == 96


def test_episode_is_truncated_not_terminated(env):
    env.reset(options={"day": 0})
    for k in range(env.episode_len):
        obs, _, term, trunc, _ = env.step(action(env))
        assert not term
        assert trunc == (k == env.episode_len - 1)
    assert env.observation_space.contains(obs)
    assert obs["physical"][0] > 0.0, "final observation carries the final SOC"


def test_legacy_terminates_with_zero_observation():
    e = make_env(legacy=True)
    e.reset()
    for _ in range(e.episode_len):
        obs, _, term, trunc, _ = e.step(action(e))
    assert term and not trunc
    assert not obs["physical"].any()


def test_legacy_reset_ignores_seed():
    e = make_env(legacy=True)
    assert e.reset(seed=1)[1]["start_idx"] == e.reset(seed=2)[1]["start_idx"] == 0


def test_dst_days_are_excluded_from_calendar_days():
    ts = pd.date_range("2025-03-28", "2025-04-02", freq="15min", tz="Europe/Athens", inclusive="left")
    base = make_signals(1).iloc[0].to_dict()
    df = pd.DataFrame([{**base, "timestamp": t} for t in ts])
    e = make_env(df)
    dates = [e.day_date(k) for k in range(e.num_days)]
    assert "2025-03-30" not in dates  # 92-step day
    assert dates == ["2025-03-28", "2025-03-29", "2025-03-31", "2025-04-01"]
    for k in range(e.num_days):
        assert e._ts[e.day_start_index(k)].hour == 0


def test_price_noise_is_seeded_and_only_affects_observations():
    info = InformationConfig(price_noise_sigma=0.3, noise_seed=11)
    signals = make_signals()
    signals["price_imb_eur_mwh"] = 60.0 + 10.0 * np.sin(np.arange(len(signals)) / 5.0)
    a, b = make_env(signals, information=info), make_env(signals, information=info)
    clean = make_env(signals)
    oa, _ = a.reset(options={"day": 1})
    ob, _ = b.reset(options={"day": 1})
    oc, _ = clean.reset(options={"day": 1})
    np.testing.assert_array_equal(oa["day_ahead_context"], ob["day_ahead_context"])
    assert not np.allclose(oa["day_ahead_context"], oc["day_ahead_context"])
    # Settlement uses realised prices: identical rewards with and without noise.
    ra = a.step(action(a, dam=50.0))[1]
    rc = clean.step(action(clean, dam=50.0))[1]
    assert ra == rc


def test_dam_price_revealed_after_commitment():
    info = InformationConfig(price_noise_sigma=0.5, noise_seed=3)
    signals = make_signals()
    signals["price_dam_eur_mwh"] = 100.0 + 20.0 * np.cos(np.arange(len(signals)) / 7.0)
    e = make_env(signals, information=info)
    obs, _ = e.reset(options={"day": 0})
    truth = signals["price_dam_eur_mwh"].to_numpy()[:96].astype(np.float32)
    before = obs["day_ahead_context"].reshape(96, 8)[:, 1]
    assert not np.allclose(before, truth)
    obs = e.step(action(e))[0]
    np.testing.assert_allclose(obs["day_ahead_context"].reshape(96, 8)[:, 1], truth)


def test_pv_forecast_error_scale():
    signals = make_signals(sunny=True)
    perfect = make_env(signals, information=InformationConfig(pv_forecast_error_scale=0.0))
    obs, _ = perfect.reset(options={"day": 0})
    ctx = obs["day_ahead_context"].reshape(96, 8)[:, 0]
    np.testing.assert_allclose(ctx, perfect._pv_act[:96], rtol=1e-5, atol=1e-4)
    real = make_env(signals, information=InformationConfig(pv_forecast_error_scale=1.0))
    obs, _ = real.reset(options={"day": 0})
    assert np.abs(obs["day_ahead_context"].reshape(96, 8)[:, 0] - real._pv_act[:96]).max() > 1.0

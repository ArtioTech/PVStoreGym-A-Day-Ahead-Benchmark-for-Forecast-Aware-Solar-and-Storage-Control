import numpy as np
import pytest

from conftest import make_env, make_signals
from pv_bess_gym.evaluation import evaluate_agent
from pv_bess_gym.oracle import OraclePlanAgent, solve_oracle


def _market():
    df = make_signals(days=2, sunny=True, act_up=0.1, act_down=0.05)
    n = np.arange(len(df))
    df["price_dam_eur_mwh"] = 100.0 + 60.0 * np.sin(2 * np.pi * n / 96.0)
    df["price_id_eur_mwh"] = df["price_dam_eur_mwh"] + 8.0 * np.cos(2 * np.pi * n / 48.0)
    df["price_imb_eur_mwh"] = df["price_dam_eur_mwh"] + 25.0 * np.sin(2 * np.pi * n / 17.0)
    return df


@pytest.fixture(scope="module")
def solved():
    env = make_env(_market(), soc_init=0.3)
    result = solve_oracle(env, [0, 1], soc_init=0.3, time_limit_s=60.0)
    return env, result


def test_replay_never_exceeds_the_bound_and_is_close(solved):
    env, result = solved
    replay = evaluate_agent(env, OraclePlanAgent(env, result), day_indices=[0, 1], soc_init=0.3)
    assert replay.total_profit <= result.upper_bound_eur + 1e-6
    assert replay.total_profit >= 0.97 * result.objective_eur
    assert replay.combined_df["violation_kw"].max() == 0.0


def test_breakdown_sums_to_objective(solved):
    _, result = solved
    b = result.breakdown
    total = (
        b["dam_eur"] + b["id_eur"] + b["imbalance_eur"] + b["afrr_capacity_eur"]
        + b["afrr_activation_eur"] - b["degradation_eur"]
    )
    assert total == pytest.approx(result.objective_eur, abs=1e-4)


def test_bound_dominates_random_and_idle_policies(solved):
    env, result = solved
    for seed in range(3):
        env.action_space.seed(seed)
        total, soc = 0.0, 0.3
        for day in [0, 1]:
            env.reset(options={"day": day, "soc_init": soc})
            for _ in range(env.episode_len):
                total += env.step(env.action_space.sample())[1]
            soc = env.battery.soc
        assert total <= result.upper_bound_eur


def test_no_imbalance_oracle_is_below_the_full_oracle(solved):
    env, result = solved
    physical = solve_oracle(env, [0, 1], soc_init=0.3, allow_imbalance=False, time_limit_s=60.0)
    assert physical.objective_eur <= result.upper_bound_eur + 1e-6
    assert abs(physical.breakdown["imbalance_eur"]) < 1e-4


def test_gate_closure_is_respected_in_the_plan(solved):
    env, result = solved
    gate = env.intraday_gate_steps
    assert np.all(result.plan["id_adjustment_kw"][:, :gate] == 0.0)


def test_oracle_rejects_legacy_env():
    with pytest.raises(ValueError):
        solve_oracle(make_env(legacy=True), [0])

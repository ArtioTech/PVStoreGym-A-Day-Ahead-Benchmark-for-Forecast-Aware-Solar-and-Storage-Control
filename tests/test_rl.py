import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from conftest import make_env, make_signals
from pv_bess_gym.evaluation import evaluate_agent
from pv_bess_gym.rl import DispatchRLWrapper, RLPolicyAgent, flatten_observation, project_setpoint, scale_action


@pytest.fixture
def base():
    return make_env(make_signals(sunny=True, act_up=0.3, act_down=0.3))


def test_wrapper_passes_check_env(base):
    check_env(DispatchRLWrapper(base), skip_render_check=True)


def test_observation_is_flat_bounded_and_matches_space(base):
    wrapped = DispatchRLWrapper(base)
    obs, _ = wrapped.reset(seed=0)
    assert obs.shape == wrapped.observation_space.shape == (24 + 8 * 9 + 1,)
    assert wrapped.observation_space.contains(obs)
    assert np.abs(obs).max() <= 10.0


def test_action_scaling(base):
    assert scale_action(base, np.array([1.0, -1.0])) == (373.0, -373.0)
    assert scale_action(base, np.array([5.0, 0.5])) == (373.0, 186.5)  # clipped to [-1, 1]


def test_wrapper_is_seed_deterministic(base):
    def rollout():
        wrapped = DispatchRLWrapper(make_env(make_signals(sunny=True, act_up=0.3, act_down=0.3)))
        wrapped.reset(seed=5)
        rng = np.random.default_rng(5)
        return [wrapped.step(rng.uniform(-1, 1, 2))[1] for _ in range(30)]

    assert rollout() == rollout()


def test_projection_respects_pcs_and_reserve(base):
    p = project_setpoint(base, 373.0, soc=0.6, cap_up_kw=100.0, cap_down_kw=50.0)
    assert p == pytest.approx(273.0, abs=2e-3)
    p = project_setpoint(base, -373.0, soc=0.6, cap_up_kw=100.0, cap_down_kw=50.0)
    assert p == pytest.approx(-323.0, abs=2e-3)


def test_projection_blocks_discharge_when_empty_and_charge_when_full(base):
    cfg = base.plant.battery
    assert project_setpoint(base, 300.0, soc=cfg.soc_min, cap_up_kw=0.0, cap_down_kw=0.0) == 0.0
    assert project_setpoint(base, -300.0, soc=cfg.soc_max, cap_up_kw=0.0, cap_down_kw=0.0) == 0.0


def test_projected_random_policy_has_no_violations_or_shortfalls(base):
    rng = np.random.default_rng(0)
    agent = RLPolicyAgent(base, lambda obs: rng.uniform(-1, 1, 2), projection=True)
    log = evaluate_agent(base, agent, day_indices=[0, 1, 2], soc_init=0.5).combined_df
    assert log["violation_kw"].max() < 1e-6
    assert log["shortfall_mwh"].sum() == pytest.approx(0.0, abs=1e-9)

    rng = np.random.default_rng(0)
    raw = RLPolicyAgent(base, lambda obs: rng.uniform(-1, 1, 2), projection=False)
    raw_log = evaluate_agent(base, raw, day_indices=[0, 1, 2], soc_init=0.5).combined_df
    assert (raw_log["violation_kw"].max() > 0) or (raw_log["shortfall_mwh"].sum() > 0)


def test_policy_agent_uses_the_same_observation_as_the_wrapper(base):
    seen = []
    agent = RLPolicyAgent(base, lambda obs: (seen.append(obs), np.zeros(2))[1])
    obs, _ = base.reset(options={"day": 0})
    agent.reset()
    agent.act(obs)
    np.testing.assert_array_equal(seen[0], flatten_observation(base, obs))

"""
PV-BESS-Gym: Gymnasium Interface
=================================

This script is the entry point for connecting new algorithms to the simulator.
It shows the full Gym contract: how to build the environment, what the
observation looks like, what the action must contain, and how to run a loop.

Run it directly to confirm your installation works:

    python scripts/gym_interface.py

Implement your own algorithm by following the MyAgent template below and
registering it in pv_bess_gym/agents/registry.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pv_bess_gym import make_unified_env  # noqa: E402


# =============================================================================
# ENVIRONMENT
# =============================================================================

def build_env(signals_path=None, plant_config_path=None):
    """Build the canonical Gym environment with default example data."""
    signals_path = signals_path or ROOT / "example_data"
    plant_config_path = plant_config_path or ROOT / "example_data" / "station_athens.json"
    return make_unified_env(
        signals_path=signals_path,
        plant_config_path=plant_config_path,
    )


# =============================================================================
# OBSERVATION REFERENCE
# =============================================================================
#
# obs is a dict with four keys:
#
# obs["physical"]  — np.ndarray shape (24,)
#   [0]  battery SOC  (0..1)
#   [1]  battery energy kWh
#   [2]  PV actual power kW
#   [3]  PV forecast power kW
#   [4]  GHI actual W/m²
#   [5]  air temperature °C
#   [6]  wind speed m/s
#   [7]  DAM price EUR/MWh
#   [8]  SIDC price EUR/MWh
#   [9]  imbalance price EUR/MWh
#   [10] aFRR up capacity price EUR/MW/h
#   [11] aFRR down capacity price EUR/MW/h
#   [12] aFRR activation up signal [0..1]
#   [13] aFRR activation down signal [0..1]
#   [14] DAM commitment at current step kW
#   [15] SIDC adjustment at current step kW
#   [16] scheduled position (DAM + SIDC) kW
#   [17] steps until next intraday block
#   [18] aFRR block steps remaining
#   [19] time-of-day sin
#   [20] time-of-day cos
#   [21] episode progress (0..1)
#   [22] D-1 committed aFRR up capacity kW   ← locked for full day
#   [23] D-1 committed aFRR down capacity kW ← locked for full day
#
# obs["forecast"]  — np.ndarray shape (horizon_steps * 9,)
#   For each future step k (default horizon = 8 steps = 2 hours):
#   [k*9+0] forecast PV kW
#   [k*9+1] GHI forecast W/m²
#   [k*9+2] temperature forecast °C
#   [k*9+3] DAM price
#   [k*9+4] SIDC price
#   [k*9+5] imbalance price
#   [k*9+6] aFRR up cap price
#   [k*9+7] aFRR down cap price
#   [k*9+8] commercial target (DAM + current SIDC) kW
#
# obs["day_ahead_context"]  — np.ndarray shape (96 * 8,)
#   Full delivery-day view, used by the D-1 optimizer.
#   For each slot t=0..95:
#   [t*8+0] forecast PV kW
#   [t*8+1] DAM price
#   [t*8+2] SIDC price
#   [t*8+3] imbalance price
#   [t*8+4] aFRR up cap price
#   [t*8+5] aFRR down cap price
#   [t*8+6] time-of-day sin
#   [t*8+7] time-of-day cos
#
# obs["action_mask"]  — np.ndarray shape (3,)
#   [0] = 1 → day_ahead + day_ahead_afrr are due (only at t=0)
#   [1] = 1 → intraday SIDC correction is due (every block boundary)
#   [2] = 1 → real_time dispatch is due (every step, always 1)


# =============================================================================
# ACTION REFERENCE
# =============================================================================
#
# action is a dict with four keys:
#
# action["day_ahead"]       shape (96,)   — DAM energy schedule kW, one per slot
#                                           required when action_mask[0] == 1
#
# action["day_ahead_afrr"]  shape (2,)    — [cap_up_kw, cap_down_kw]
#                                           D-1 aFRR capacity offer, fixed for day
#                                           required when action_mask[0] == 1
#
# action["intraday"]        shape (1,)    — [id_adjustment_kw]
#                                           additive SIDC energy correction
#                                           required when action_mask[1] == 1
#
# action["real_time"]       shape (1,)    — [p_batt_kw]
#                                           battery dispatch setpoint
#                                           positive = discharge, negative = charge
#                                           required every step


# =============================================================================
# ALGORITHM TEMPLATE
# =============================================================================

class MyAgent:
    """Minimal agent template.

    Replace the random sampling in act() with your own logic.
    The only contract is:
        agent.reset()         — called at the start of each episode
        agent.act(obs) → action  — called at every simulator step
    """

    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def reset(self) -> None:
        """Called at the start of each episode. Initialize any internal state."""
        pass

    def act(self, obs: dict) -> dict:
        """Return an action dict given the current observation.

        Use obs["action_mask"] to know which keys are required this step.
        Keys not required this step are ignored by the environment.
        """
        action_mask = obs["action_mask"]
        action = {
            "day_ahead":      np.zeros(self.env.episode_len, dtype=np.float32),
            "day_ahead_afrr": np.zeros(2, dtype=np.float32),
            "intraday":       np.zeros(1, dtype=np.float32),
            "real_time":      np.zeros(1, dtype=np.float32),
        }

        if action_mask[0] > 0.5:
            # D-1: commit DAM schedule and aFRR capacity for the full day.
            # Replace with your D-1 optimizer or RL policy output.
            action["day_ahead"] = self.action_space["day_ahead"].sample()
            action["day_ahead_afrr"] = self.action_space["day_ahead_afrr"].sample()

        if action_mask[1] > 0.5:
            # Intraday: SIDC energy correction for the upcoming block.
            # Replace with your intraday correction logic.
            action["intraday"] = self.action_space["intraday"].sample()

        # Real-time: battery dispatch setpoint, every step.
        # Replace with your real-time controller.
        action["real_time"] = self.action_space["real_time"].sample()

        return action


# =============================================================================
# REGISTRATION
# =============================================================================
#
# To make your agent available to run_benchmarks.py, register it in:
#   pv_bess_gym/agents/registry.py
#
# Example:
#
#   from pv_bess_gym.agents.registry import ALGORITHM_REGISTRY, AlgorithmSpec
#
#   def _build_my_agent(env, options):
#       return MyAgent(env)
#
#   ALGORITHM_REGISTRY["my-agent"] = AlgorithmSpec(
#       name="my-agent",
#       market_mode="unified",
#       description="My custom controller.",
#       builder=_build_my_agent,
#       benchmark=False,
#       include_in_all=False,
#   )
#
# Then run it:
#   python scripts/run_benchmarks.py --algorithm my-agent --num-days 3


# =============================================================================
# DEMO LOOP
# =============================================================================

def run_demo(num_days: int = 1) -> None:
    """Run one full episode with the random MyAgent and print a summary."""
    env = build_env()
    agent = MyAgent(env)

    print("Environment built.")
    print(f"  Action space:      {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Episode length:    {env.episode_len} steps ({env.market.dt_h * 60:.0f}-min resolution)")
    print()

    steps_per_day = int(round(24.0 / env.market.dt_h))

    for day in range(num_days):
        agent.reset()
        obs, info = env.reset(options={"start_idx": day * steps_per_day, "soc_init": 0.50})

        total_reward = 0.0
        step_count = 0
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1

        print(f"Day {day:02d} | steps={step_count} | profit={total_reward:.2f} EUR | final_soc={env.battery.soc:.3f}")

    print("\nDone. Replace MyAgent.act() with your algorithm to run real experiments.")


if __name__ == "__main__":
    run_demo(num_days=1)

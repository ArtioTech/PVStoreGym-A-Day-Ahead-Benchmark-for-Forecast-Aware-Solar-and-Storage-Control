"""Tests that need the built dataset (data/processed); skipped when it is absent."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

import pv_bess_gym  # noqa: F401  (registers the task IDs)
from pv_bess_gym.benchmark.tasks import TASK_NAMES, Task, make_task_env, task_env_id
from test_brief_units import audit_day

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(
    not (DATA / "publication_times_2025.csv").exists(), reason="built dataset with sidecar files not present"
)
TEST_ONLY_C_THR = 1.0   # EUR/MWh, only so that T4/T5 can be constructed for the API check; not a benchmark value


@pytest.mark.parametrize("name", TASK_NAMES)
def test_check_env_on_every_registered_task(name):
    kwargs = {"c_thr_eur_mwh": TEST_ONLY_C_THR} if Task.load(name).c_thr_eur_mwh is None else {}
    env = gym.make(task_env_id(name), **kwargs).unwrapped
    check_env(env, skip_render_check=True)


def test_ageing_tasks_refuse_to_run_without_the_author_value():
    for name in TASK_NAMES:
        if Task.load(name).c_thr_eur_mwh is None:
            with pytest.raises(ValueError, match="author-supplied"):
                make_task_env(name)


@pytest.mark.parametrize("name", ["T1-Nominal", "T2-Forecast"])
def test_U7_full_year_observation_audit(name):
    """No price in the observation before its publication time, every step of every day of 2025."""
    env = make_task_env(name)
    pub = pd.read_csv(DATA / "publication_times_2025.csv")          # read independently of the env
    pub["published_at"] = pd.to_datetime(pub["published_at"], utc=True)
    table = {(row.delivery_date, row.market): row.published_at for row in pub.itertuples(index=False)}
    label_market = {"dam": "DAM", "ida1": "IDA1", "ida2": "IDA2", "ida3": "IDA3"}
    checked = 0
    for day in range(env.num_days):
        start = env._ts[env.day_start_index(day)]
        date = start.date().isoformat()

        def pub_step(label, _start=start, _date=date):
            if label == "dam_forecast":
                return -(10**9)
            minutes = (table[(_date, label_market[label])] - _start.tz_convert("UTC")).total_seconds() / 60.0
            return int(np.ceil(minutes / 15.0))

        imb = env._imb[env.day_start_index(day): env.day_start_index(day) + env.episode_len]
        checked += audit_day(env, day, pub_step, imb)
    assert checked == env.num_days * 96 * 96 * 3

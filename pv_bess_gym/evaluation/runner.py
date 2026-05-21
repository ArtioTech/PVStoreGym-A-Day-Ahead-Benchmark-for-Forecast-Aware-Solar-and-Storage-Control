"""Core evaluation loop for PV+BESS Gym benchmark runs.

This module provides the minimal surface needed to evaluate any agent against
the UnifiedPVBESSGreeceEnv:

    evaluate_agent()          - run an agent over one or more dataset days
    save_evaluation_outputs() - persist CSV logs and plots
    EvaluationConfig          - parameters dataclass
    EvaluationResult          - structured result dataclass
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

import pandas as pd

from pv_bess_gym.utils.viz import plot_episode


class AgentLike(Protocol):
    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        ...

    def reset(self) -> None:
        ...


@dataclass
class EvaluationConfig:
    """Parameters for a benchmark evaluation run."""

    signals_path: str | Path
    plant_config_path: str | Path
    output_dir: Optional[str | Path] = None
    start_day: int = 0
    num_days: int = 1
    day_indices: Optional[Sequence[int]] = None
    carry_soc_between_days: bool = True
    soc_init: float = 0.10
    tz: Optional[str] = "Europe/Athens"
    dt_hours: float = 0.25
    afrr_block_minutes: int = 60
    decision_block_minutes: int = 60
    forecast_horizon_steps: int = 8
    violation_penalty: float = 100.0
    shortfall_penalty: float = 20.0
    include_forecasts: bool = True
    title: str = "PV+BESS benchmark"
    agent_name: str = "da-opt-rbc"


@dataclass
class EvaluationResult:
    """Structured outputs from a completed evaluation run."""

    day_indices: list[int]
    total_profit: float
    final_soc: float
    combined_log: list[dict[str, Any]]
    daily_summary: list[dict[str, Any]]
    combined_df: pd.DataFrame
    summary_df: pd.DataFrame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_day_indices(
    *,
    available_days: int,
    start_day: int,
    num_days: int,
    day_indices: Optional[Sequence[int]],
) -> list[int]:
    if day_indices is not None:
        resolved = [int(d) for d in day_indices]
        if not resolved:
            raise ValueError("day_indices must not be empty when provided.")
    else:
        if start_day < 0 or start_day >= available_days:
            raise ValueError(f"start_day must be between 0 and {available_days - 1}.")
        if num_days < 1:
            raise ValueError("num_days must be >= 1.")
        if start_day + num_days > available_days:
            raise ValueError(
                f"Requested {num_days} day(s) from start day {start_day}, "
                f"but only {available_days - start_day} full day(s) are available."
            )
        resolved = list(range(start_day, start_day + num_days))

    invalid = [d for d in resolved if d < 0 or d >= available_days]
    if invalid:
        raise ValueError(f"Day indices out of range 0..{available_days - 1}: {invalid}")
    return resolved


def _agent_reset(agent: Any) -> None:
    reset_fn = getattr(agent, "reset", None)
    if callable(reset_fn):
        reset_fn()


def _agent_act(agent: Any, obs: dict[str, Any]) -> dict[str, Any]:
    act_fn = getattr(agent, "act", None)
    if callable(act_fn):
        return act_fn(obs)
    if callable(agent):
        return agent(obs)
    raise TypeError("Agent must expose an .act(obs) method or be directly callable.")


def _run_single_day(
    env: Any,
    agent: Any,
    *,
    start_idx: int,
    soc_init: float,
    day_label: int,
    global_step_offset: int,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Run one episode and return (profit, final_soc, step_log)."""
    _agent_reset(agent)
    obs, reset_info = env.reset(options={"start_idx": start_idx, "soc_init": soc_init})

    done = False
    day_profit = 0.0
    day_log: list[dict[str, Any]] = []

    while not done:
        action = _agent_act(agent, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        day_profit += reward

        row = dict(info)
        row["day_index"] = day_label
        row["episode_start_idx"] = reset_info["start_idx"]
        row["global_step"] = global_step_offset + int(info["t"])
        day_log.append(row)

    return float(day_profit), float(env.battery.soc), day_log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_agent(
    env: Any,
    agent: Any,
    *,
    start_day: int = 0,
    num_days: int = 1,
    day_indices: Optional[Sequence[int]] = None,
    carry_soc_between_days: bool = True,
    soc_init: float = 0.10,
) -> EvaluationResult:
    """Evaluate an agent over one or more consecutive dataset days.

    Parameters
    ----------
    env:
        A built UnifiedPVBESSGreeceEnv (or any compatible Gym env).
    agent:
        Any object exposing ``reset()`` and ``act(obs) -> action``.
    start_day:
        First dataset day index (ignored if day_indices is provided).
    num_days:
        Number of consecutive days (ignored if day_indices is provided).
    day_indices:
        Explicit list of dataset day indices. Overrides start_day / num_days.
    carry_soc_between_days:
        If True, the battery SOC at the end of each day is carried into the next.
    soc_init:
        Initial SOC for the first evaluated day (and for all days if not carrying).

    Returns
    -------
    EvaluationResult with per-step logs, daily summary, and totals.
    """
    steps_per_day = int(round(24.0 / env.market.dt_h))
    available_days = len(env.df_all) // steps_per_day
    resolved_days = _resolve_day_indices(
        available_days=available_days,
        start_day=start_day,
        num_days=num_days,
        day_indices=day_indices,
    )

    current_soc = soc_init
    all_logs: list[dict[str, Any]] = []
    daily_summary: list[dict[str, Any]] = []

    for sequence_index, dataset_day in enumerate(resolved_days):
        day_start_idx = dataset_day * steps_per_day
        day_soc_init = current_soc if (carry_soc_between_days or sequence_index == 0) else soc_init
        day_profit, current_soc, day_log = _run_single_day(
            env=env,
            agent=agent,
            start_idx=day_start_idx,
            soc_init=day_soc_init,
            day_label=dataset_day,
            global_step_offset=sequence_index * steps_per_day,
        )
        all_logs.extend(day_log)
        daily_summary.append({
            "day_index": dataset_day,
            "start_idx": day_start_idx,
            "start_timestamp": day_log[0]["timestamp"] if day_log else None,
            "end_timestamp": day_log[-1]["timestamp"] if day_log else None,
            "profit_eur": day_profit,
            "final_soc": current_soc,
        })

    combined_df = pd.DataFrame(all_logs)
    summary_df = pd.DataFrame(daily_summary)
    total_profit = float(sum(item["profit_eur"] for item in daily_summary))

    return EvaluationResult(
        day_indices=resolved_days,
        total_profit=total_profit,
        final_soc=float(current_soc),
        combined_log=all_logs,
        daily_summary=daily_summary,
        combined_df=combined_df,
        summary_df=summary_df,
    )


def save_evaluation_outputs(
    result: EvaluationResult,
    *,
    output_dir: str | Path,
    title: str,
) -> None:
    """Save CSV logs and diagnostic plots for a completed evaluation.

    Writes:
        combined_log.csv   - one row per simulator step
        daily_summary.csv  - one row per evaluated day
        power.png, soc.png, profit.png, revenue_stack.png
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result.combined_df.to_csv(output_path / "combined_log.csv", index=False)
    result.summary_df.to_csv(output_path / "daily_summary.csv", index=False)

    running_profit = 0.0
    plot_log: list[dict[str, Any]] = []
    for row in result.combined_log:
        row_copy = dict(row)
        row_copy["t"] = row_copy["global_step"]
        running_profit += float(row_copy["reward"])
        row_copy["cum_profit"] = running_profit
        plot_log.append(row_copy)

    if plot_log:
        plot_episode(plot_log, title=title, out_dir=str(output_path))

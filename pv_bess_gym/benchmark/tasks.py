"""Benchmark tasks T1-T5 and the journal protocol (configs/journal.yaml = rev1; v3 and v4 in configs/journal_v3.yaml, configs/journal_v4.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from pv_bess_gym.envs.factory import make_unified_env

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = REPO_ROOT / "configs" / "tasks"
TASK_NAMES = ["T1-Nominal", "T2-Forecast", "T3-Gate", "T4-Ageing", "T5-Combined"]
PROTOCOL_PATH = REPO_ROOT / "configs" / "journal.yaml"
PROTOCOL_ENV = "PVBESS_PROTOCOL"     # path of another protocol, e.g. configs/journal_v3.yaml


def task_env_id(name: str) -> str:
    return f"PVBESS-{name}-v2"


@dataclass(frozen=True)
class Task:
    name: str
    forecast_residual_scaling: float
    da_price_info: str
    gate_closure_minutes: int
    c_thr_eur_mwh: Optional[float]

    @classmethod
    def load(cls, name: str) -> "Task":
        return cls(**yaml.safe_load((TASK_DIR / f"{name}.yaml").read_text(encoding="utf-8")))


def protocol_path() -> Path:
    """The active protocol: $PVBESS_PROTOCOL if set, else configs/journal.yaml."""
    import os

    value = os.environ.get(PROTOCOL_ENV)
    if not value:
        return PROTOCOL_PATH
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_protocol(path: Optional[str | Path] = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or protocol_path()).read_text(encoding="utf-8"))


def results_root(protocol: Optional[dict[str, Any]] = None) -> Path:
    """Directory of the results of the active protocol (``results/<results_subdir>``)."""
    proto = protocol or load_protocol()
    return REPO_ROOT / "results" / str(proto.get("results_subdir", "") or "")


def figures_root(protocol: Optional[dict[str, Any]] = None) -> Path:
    proto = protocol or load_protocol()
    return REPO_ROOT / "figures" / str(proto.get("results_subdir", "") or "")


def make_task_env(
    task: str | Task = "T1-Nominal",
    *,
    signals_path: Optional[str | Path] = None,
    plant_config_path: Optional[str | Path] = None,
    c_thr_eur_mwh: Optional[float] = None,
    episode_days: Optional[Sequence[int]] = None,
    protocol: Optional[dict[str, Any]] = None,
    **overrides: Any,
):
    """Build the environment of a benchmark task with the journal protocol settings.

    T4-Ageing and T5-Combined need the degradation cost ``c_thr_eur_mwh``; it
    is an author-supplied value and is not set in the task files, so it must be
    passed here until it is.
    """
    spec = task if isinstance(task, Task) else Task.load(task)
    proto = protocol or load_protocol()
    sim = proto["simulator"]
    task_over = dict((proto.get("task_overrides") or {}).get(spec.name, {}))
    c_thr = spec.c_thr_eur_mwh if c_thr_eur_mwh is None else c_thr_eur_mwh
    if c_thr is None:
        raise ValueError(
            f"Task {spec.name} needs c_thr_eur_mwh, an author-supplied value that has not been set. "
            f"Set it in configs/tasks/{spec.name}.yaml or pass c_thr_eur_mwh=..."
        )
    kwargs: dict[str, Any] = dict(
        signals_path=signals_path or REPO_ROOT / proto["data"]["signals"],
        plant_config_path=plant_config_path or REPO_ROOT / proto["data"]["plant"],
        soc_init=float(proto["episode"]["soc_init"]),
        settlement=sim["settlement"],
        metering_losses=bool(sim["metering_losses"]),
        battery_aux_drains=bool(sim["battery_aux_drains"]),
        directional_taper=bool(sim["directional_taper"]),
        energy_limited_power=bool(sim["energy_limited_power"]),
        calendar_days=bool(sim["calendar_days"]),
        info_filter=bool(sim["info_filter"]),
        da_forecaster=sim["da_forecaster"],
        da_price_info=task_over.pop("da_price_info", spec.da_price_info),
        da_weather=sim.get("da_weather", "short_lead"),
        id_weather=sim.get("id_weather", "same_as_da"),
        short_lead_visible_h=float(sim.get("short_lead_visible_h", 3.0)),
        da_plant_consistent=bool(sim.get("da_plant_consistent", False)),
        market_clock=str(sim.get("market_clock", "as_published")),
        radiation_stamp=str(sim.get("radiation_stamp", "as_published")),
        id_reference=str(sim.get("id_reference", "pv_only")),
        id_position_limit=str(sim.get("id_position_limit", "off")),
        pv_forecast_error_scale=float(spec.forecast_residual_scaling),
        intraday_gate_closure_minutes=int(spec.gate_closure_minutes),
        c_thr_eur_mwh=float(c_thr),
        episode_days=episode_days,
    )
    if task_over:
        raise ValueError(f"Unknown task_overrides for {spec.name}: {sorted(task_over)}")
    kwargs.update(overrides)
    return make_unified_env(**kwargs)


def heldout_days(env: Any, protocol: Optional[dict[str, Any]] = None) -> list[int]:
    """Days of the within-regime held-out weeks (inside the train range, never trained on)."""
    import datetime as _dt

    proto = protocol or load_protocol()
    weeks = {int(w) for w in proto.get("heldout_iso_weeks", [])}
    start, end = (str(v) for v in proto["split"]["train"])
    out = []
    for k in range(env.num_days):
        date = env.day_date(k)
        if start <= date <= end and _dt.date.fromisoformat(date).isocalendar()[1] in weeks:
            out.append(k)
    return out


def split_days(env: Any, split: str, protocol: Optional[dict[str, Any]] = None) -> list[int]:
    """Day indices of a split: ``train`` (without the held-out weeks), ``validation``,
    ``test`` or ``heldout`` (the within-regime weeks)."""
    proto = protocol or load_protocol()
    held = heldout_days(env, proto)
    if split == "heldout":
        return held
    start, end = (str(v) for v in proto["split"][split])
    days = [k for k in range(env.num_days) if start <= env.day_date(k) <= end]
    return [k for k in days if k not in set(held)] if split == "train" else days

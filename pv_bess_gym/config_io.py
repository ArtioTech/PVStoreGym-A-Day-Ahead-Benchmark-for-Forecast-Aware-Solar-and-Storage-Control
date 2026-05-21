from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from pv_bess_gym.envs import PlantConfig
from pv_bess_gym.models.battery import BatteryConfig
from pv_bess_gym.models.grid import GridConfig
from pv_bess_gym.models.pv import PVConfig


def load_plant_config(path: str | Path) -> PlantConfig:
    """Load PV, battery, and grid properties from a JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlantConfig(
        pv=PVConfig(**payload["pv"]),
        battery=BatteryConfig(**payload["battery"]),
        grid=GridConfig(**payload["grid"]),
    )


def save_plant_config(path: str | Path, plant: PlantConfig) -> None:
    """Save a plant configuration to JSON."""
    Path(path).write_text(
        json.dumps(
            {
                "pv": asdict(plant.pv),
                "battery": asdict(plant.battery),
                "grid": asdict(plant.grid),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

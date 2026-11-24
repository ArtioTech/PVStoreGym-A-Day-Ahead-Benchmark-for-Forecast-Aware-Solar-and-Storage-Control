"""PV-BESS-Gym: Gymnasium simulator for PV+BESS Greek market operation."""

__version__ = "2.1.0"

from pv_bess_gym.config_io import load_plant_config, save_plant_config
from pv_bess_gym.data.signals import SignalSchema, load_signals
from pv_bess_gym.agents import (
    ALGORITHM_REGISTRY,
    DayAheadMPCConfig,
    build_registered_algorithm,
    default_algorithm_names,
    get_algorithm_spec,
    list_algorithm_specs,
)
from pv_bess_gym.envs import (
    DayAheadMarketTiming,
    InformationConfig,
    MarketConfig,
    SimOptions,
    PlantConfig,
    UnifiedPVBESSGreeceEnv,
    make_unified_env,
)
from pv_bess_gym.evaluation import (
    EvaluationConfig,
    EvaluationResult,
    evaluate_agent,
    save_evaluation_outputs,
)

from gymnasium.envs.registration import register, registry

if "PVBESSGreece-v1" not in registry:
    # gym.make("PVBESSGreece-v1", signals_path=..., plant_config_path=...)
    register(
        id="PVBESSGreece-v1",
        entry_point="pv_bess_gym.envs.factory:make_unified_env",
        disable_env_checker=True,
        order_enforce=False,
    )

# Benchmark tasks: gym.make("PVBESS-T1-Nominal-v2"), ... "PVBESS-T5-Combined-v2".
from pv_bess_gym.benchmark.tasks import TASK_NAMES, make_task_env, task_env_id  # noqa: E402

for _task in TASK_NAMES:
    if task_env_id(_task) not in registry:
        register(
            id=task_env_id(_task),
            entry_point="pv_bess_gym.benchmark.tasks:make_task_env",
            kwargs={"task": _task},
            disable_env_checker=True,
            order_enforce=False,
        )

__all__ = [
    "__version__",
    "TASK_NAMES",
    "make_task_env",
    "task_env_id",
    # Environment
    "UnifiedPVBESSGreeceEnv",
    "make_unified_env",
    "DayAheadMarketTiming",
    "InformationConfig",
    "SimOptions",
    "MarketConfig",
    "PlantConfig",
    # Data
    "SignalSchema",
    "load_signals",
    "load_plant_config",
    "save_plant_config",
    # Agents & registry
    "ALGORITHM_REGISTRY",
    "DayAheadMPCConfig",
    "build_registered_algorithm",
    "default_algorithm_names",
    "get_algorithm_spec",
    "list_algorithm_specs",
    # Evaluation
    "EvaluationConfig",
    "EvaluationResult",
    "evaluate_agent",
    "save_evaluation_outputs",
]

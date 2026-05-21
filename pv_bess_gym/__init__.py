"""PV-BESS-Gym: Gymnasium simulator for PV+BESS Greek market operation."""

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
    MarketConfig,
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

__all__ = [
    # Environment
    "UnifiedPVBESSGreeceEnv",
    "make_unified_env",
    "DayAheadMarketTiming",
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

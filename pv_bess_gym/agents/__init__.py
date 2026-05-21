"""Benchmark agents and algorithm registry for the PV+BESS Gym simulator.

Primary agents (target UnifiedPVBESSGreeceEnv):
    DayAheadOptimizedDARBCAgent       - D-1 MILP + RBC intraday/real-time
    DayAheadOptimizedDAMPCIntradayRealTimeAgent - D-1 MILP + MPC intraday/real-time
    DayAheadOptimizedDARTMPCAgent     - D-1 MILP + RBC intraday + MPC real-time

Legacy agents (target HierarchicalPVBESSGreeceEnv, kept for backward compatibility):
    See pv_bess_gym/agents/legacy/
"""

from pv_bess_gym.agents.day_ahead_baselines import (
    DayAheadMPCConfig,
    DayAheadOptimizedDAMPCIntradayRealTimeAgent,
    DayAheadOptimizedDARBCAgent,
    DayAheadOptimizedDARTMPCAgent,
)
from pv_bess_gym.agents.registry import (
    ALGORITHM_REGISTRY,
    AlgorithmSpec,
    build_registered_algorithm,
    default_algorithm_names,
    get_algorithm_spec,
    list_algorithm_specs,
    normalize_algorithm_name,
)

__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmSpec",
    "DayAheadMPCConfig",
    "DayAheadOptimizedDAMPCIntradayRealTimeAgent",
    "DayAheadOptimizedDARBCAgent",
    "DayAheadOptimizedDARTMPCAgent",
    "build_registered_algorithm",
    "default_algorithm_names",
    "get_algorithm_spec",
    "list_algorithm_specs",
    "normalize_algorithm_name",
]

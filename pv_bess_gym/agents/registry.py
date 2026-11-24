from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from pv_bess_gym.agents.day_ahead_baselines import (
    DayAheadMPCConfig,
    DayAheadOnlyAgent,
    DayAheadOptimizedDAIDMPCAgent,
    DayAheadOptimizedDAMPCIntradayRealTimeAgent,
    DayAheadOptimizedDARBCAgent,
    DayAheadOptimizedDARTMPCAgent,
)
from pv_bess_gym.agents.legacy.hierarchical_baseline import HierarchicalHeuristicAgent
from pv_bess_gym.agents.legacy.hierarchical_mpc import HierarchicalMPCConfig, HierarchicalMPCAgent

MarketMode = Literal["unified", "rolling"]
AgentBuilder = Callable[[Any, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class AlgorithmSpec:
    """Registry entry for one simulator algorithm/controller.

    Add future controllers here instead of editing the simulator front script.
    Each entry states which environment it needs, how it is built, and whether
    it belongs in the default benchmark comparison.
    """

    name: str
    market_mode: MarketMode
    description: str
    builder: AgentBuilder
    benchmark: bool = True
    include_in_all: bool = True


def _build_da_opt_rbc(env: Any, options: Mapping[str, Any]) -> DayAheadOptimizedDARBCAgent:
    return DayAheadOptimizedDARBCAgent(env, config=options.get("day_ahead_config"))


def _build_da_opt_rt_mpc(env: Any, options: Mapping[str, Any]) -> DayAheadOptimizedDARTMPCAgent:
    return DayAheadOptimizedDARTMPCAgent(env, config=options.get("day_ahead_config"))


def _build_da_opt_mpc(env: Any, options: Mapping[str, Any]) -> DayAheadOptimizedDAMPCIntradayRealTimeAgent:
    return DayAheadOptimizedDAMPCIntradayRealTimeAgent(env, config=options.get("day_ahead_config"))


def _build_da_only(env: Any, options: Mapping[str, Any]) -> DayAheadOnlyAgent:
    return DayAheadOnlyAgent(env, config=options.get("day_ahead_config"))


def _build_da_opt_id_mpc(env: Any, options: Mapping[str, Any]) -> DayAheadOptimizedDAIDMPCAgent:
    return DayAheadOptimizedDAIDMPCAgent(env, config=options.get("day_ahead_config"))


def _build_rolling_rbc(env: Any, options: Mapping[str, Any]) -> HierarchicalHeuristicAgent:
    return HierarchicalHeuristicAgent(
        action_space=env.action_space,
        p_pcs_kw=env.plant.battery.p_pcs_kw,
        reserve_fraction=float(options.get("reserve_fraction", 0.20)),
        low_price=float(options.get("low_price", 45.0)),
        high_price=float(options.get("high_price", 95.0)),
        reserve_price_threshold=float(options.get("reserve_price_threshold", 20.0)),
    )


def _build_rolling_mpc(env: Any, options: Mapping[str, Any]) -> HierarchicalMPCAgent:
    return HierarchicalMPCAgent(
        action_space=env.action_space,
        battery_cfg=env.plant.battery,
        grid_cfg=env.plant.grid,
        dt_hours=env.market.dt_h,
        decision_block_minutes=env.hierarchy.decision_block_minutes,
        config=options.get("hierarchical_mpc_config"),
    )


ALGORITHM_REGISTRY: dict[str, AlgorithmSpec] = {
    "da-opt-rbc": AlgorithmSpec(
        name="da-opt-rbc",
        market_mode="unified",
        description="Unified benchmark: D-1 co-optimized DAM schedule and aFRR capacity, RBC SIDC intraday correction, RBC real-time battery dispatch.",
        builder=_build_da_opt_rbc,
        benchmark=True,
        include_in_all=True,
    ),
    "da-opt-mpc": AlgorithmSpec(
        name="da-opt-mpc",
        market_mode="unified",
        description="Unified benchmark: D-1 co-optimized DAM schedule and aFRR capacity, MPC SIDC intraday correction, MPC real-time battery dispatch.",
        builder=_build_da_opt_mpc,
        benchmark=True,
        include_in_all=True,
    ),
    "da-opt-rt-mpc": AlgorithmSpec(
        name="da-opt-rt-mpc",
        market_mode="unified",
        description="Ablation: D-1 co-optimized DAM schedule and aFRR capacity, RBC SIDC intraday correction, MPC real-time battery dispatch.",
        builder=_build_da_opt_rt_mpc,
        benchmark=True,
        include_in_all=False,
    ),
    "da-only": AlgorithmSpec(
        name="da-only",
        market_mode="unified",
        description="Reference: D-1 co-optimized DAM schedule and aFRR capacity, no intraday correction, RBC real-time battery dispatch.",
        builder=_build_da_only,
        benchmark=True,
        include_in_all=False,
    ),
    "da-opt-id-mpc": AlgorithmSpec(
        name="da-opt-id-mpc",
        market_mode="unified",
        description="Ablation: D-1 co-optimized DAM schedule and aFRR capacity, MPC SIDC intraday correction, RBC real-time battery dispatch.",
        builder=_build_da_opt_id_mpc,
        benchmark=True,
        include_in_all=False,
    ),
    "rolling-rbc": AlgorithmSpec(
        name="rolling-rbc",
        market_mode="rolling",
        description="Legacy rolling rule-based controller.",
        builder=_build_rolling_rbc,
        benchmark=True,
        include_in_all=True,
    ),
    "rolling-mpc": AlgorithmSpec(
        name="rolling-mpc",
        market_mode="rolling",
        description="Legacy rolling hierarchical MPC controller.",
        builder=_build_rolling_mpc,
        benchmark=True,
        include_in_all=True,
    ),
}


ALIASES: dict[str, str] = {
    "rbc": "da-opt-rbc",
    "heuristic": "da-opt-rbc",
    "mpc": "da-opt-mpc",
    "day-ahead": "da-opt-mpc",
}


def normalize_algorithm_name(name: str) -> str:
    normalized = name.strip().lower().replace("_", "-")
    return ALIASES.get(normalized, normalized)


def get_algorithm_spec(name: str) -> AlgorithmSpec:
    normalized = normalize_algorithm_name(name)
    if normalized not in ALGORITHM_REGISTRY:
        available = ", ".join(sorted(ALGORITHM_REGISTRY))
        raise ValueError(f"Unknown algorithm '{name}'. Available algorithms: {available}")
    return ALGORITHM_REGISTRY[normalized]


def build_registered_algorithm(name: str, env: Any, options: Mapping[str, Any] | None = None) -> Any:
    spec = get_algorithm_spec(name)
    return spec.builder(env, options or {})


def list_algorithm_specs(*, market_mode: MarketMode | None = None) -> list[AlgorithmSpec]:
    specs = list(ALGORITHM_REGISTRY.values())
    if market_mode is not None:
        specs = [spec for spec in specs if spec.market_mode == market_mode]
    return sorted(specs, key=lambda spec: (spec.market_mode, spec.name))


def default_algorithm_names(market_mode: MarketMode) -> list[str]:
    return [
        spec.name
        for spec in list_algorithm_specs(market_mode=market_mode)
        if spec.include_in_all
    ]


__all__ = [
    "ALGORITHM_REGISTRY",
    "AlgorithmSpec",
    "DayAheadMPCConfig",
    "HierarchicalMPCConfig",
    "build_registered_algorithm",
    "default_algorithm_names",
    "get_algorithm_spec",
    "list_algorithm_specs",
    "normalize_algorithm_name",
]

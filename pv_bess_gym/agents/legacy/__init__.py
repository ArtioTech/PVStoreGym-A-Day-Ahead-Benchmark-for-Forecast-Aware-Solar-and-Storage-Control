"""Legacy rolling-horizon agents kept for backward compatibility.

These controllers target the older HierarchicalPVBESSGreeceEnv interface.
New algorithms should target UnifiedPVBESSGreeceEnv instead and be
registered in pv_bess_gym/agents/registry.py.
"""

from pv_bess_gym.agents.legacy.hierarchical_baseline import HierarchicalHeuristicAgent
from pv_bess_gym.agents.legacy.hierarchical_mpc import HierarchicalMPCAgent, HierarchicalMPCConfig

__all__ = [
    "HierarchicalHeuristicAgent",
    "HierarchicalMPCAgent",
    "HierarchicalMPCConfig",
]

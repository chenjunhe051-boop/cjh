"""Compatibility exports for the canonical hybrid MoE family.

Historically this module duplicated the implementations in ``gated.py``.
Aliases keep old imports and serialized checkpoints loadable while ensuring
that fixes are made in one implementation only.
"""

from .gated import (
    AdaptiveBalanceController,
    AdaptiveGateMoE,
    ContextRefinedLowRankHybridAdaptiveGateMoE,
    DetailAwareLowRankHybridAdaptiveGateMoE,
    FusedAdaptiveGateMoE,
    HybridAdaptiveGateMoE,
    HybridAdaptiveGateMoEv2,
    LowRankHybridAdaptiveGateMoE,
    OptimalHybridGateMoE,
    PyramidContextMixer,
    RefinedLowRankHybridAdaptiveGateMoE,
    VisualDetailGate,
    VisualEnhancedAdaptiveGateMoE,
    _run_visual_hybrid_moe_forward,
)

__all__ = (
    "AdaptiveGateMoE",
    "VisualDetailGate",
    "PyramidContextMixer",
    "FusedAdaptiveGateMoE",
    "HybridAdaptiveGateMoE",
    "HybridAdaptiveGateMoEv2",
    "LowRankHybridAdaptiveGateMoE",
    "RefinedLowRankHybridAdaptiveGateMoE",
    "DetailAwareLowRankHybridAdaptiveGateMoE",
    "ContextRefinedLowRankHybridAdaptiveGateMoE",
    "VisualEnhancedAdaptiveGateMoE",
    "AdaptiveBalanceController",
    "OptimalHybridGateMoE",
    "_run_visual_hybrid_moe_forward",
)

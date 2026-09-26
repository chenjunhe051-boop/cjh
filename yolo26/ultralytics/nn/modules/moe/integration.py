"""Compatibility exports for canonical integrated MoE implementations."""

from .modules import (
    CrossPathGate,
    DiversifiedExpertGroup,
    DiversifiedExpertMoE,
    GatedFusionMoE,
    HyperUltimateMoE,
    MatMulFusedExperts,
    MultiHeadRouterMoE,
    MultiHeadRouterV3,
    UltimateOptimizedMoE,
    UltraLightRouter,
)

__all__ = (
    "MultiHeadRouterV3",
    "DiversifiedExpertGroup",
    "CrossPathGate",
    "MultiHeadRouterMoE",
    "DiversifiedExpertMoE",
    "GatedFusionMoE",
    "UltraLightRouter",
    "MatMulFusedExperts",
    "HyperUltimateMoE",
    "UltimateOptimizedMoE",
)

"""Compatibility exports for canonical gated MoE blocks."""

from .gated import AdaptiveGateMoE, HyperFusedMoE, HyperSplitMoE

__all__ = ("AdaptiveGateMoE", "HyperSplitMoE", "HyperFusedMoE")

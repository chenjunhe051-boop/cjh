"""Compatibility exports for canonical advanced MoE routers."""

from .gated import DualStreamGateRouter, DualStreamGateRouterV2, ZeroCostRouter

__all__ = ("DualStreamGateRouter", "DualStreamGateRouterV2", "ZeroCostRouter")

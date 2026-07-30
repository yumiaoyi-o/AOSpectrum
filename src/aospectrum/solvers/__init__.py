"""Numerical solver contracts and shared quality checks."""

from .catalog import (
    BackendDescriptor,
    backend_descriptors,
    create_band_solver,
    create_orbital_solver,
)
from .quality import assess_quality, generalized_residual_metrics

__all__ = [
    "BackendDescriptor",
    "assess_quality",
    "backend_descriptors",
    "create_band_solver",
    "create_orbital_solver",
    "generalized_residual_metrics",
]

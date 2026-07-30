"""Spectrum selections, quality evidence and solver receipts."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from aospectrum.errors import ContractError

from ._arrays import frozen_array


@dataclass(frozen=True, slots=True)
class EnergyInterval:
    """One explicit open interval in eV."""

    lower_ev: float
    upper_ev: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lower_ev) or not math.isfinite(self.upper_ev):
            raise ContractError("energy interval endpoints must be finite")
        if self.lower_ev >= self.upper_ev:
            raise ContractError("energy interval requires lower_ev < upper_ev")

    def expanded(self, guard_ev: float) -> "EnergyInterval":
        if not math.isfinite(guard_ev) or guard_ev < 0.0:
            raise ContractError("energy interval guard must be finite and nonnegative")
        return EnergyInterval(self.lower_ev - guard_ev, self.upper_ev + guard_ev)


@dataclass(frozen=True, slots=True)
class AbsoluteStateRange:
    """One-based inclusive absolute state range."""

    first: int
    last: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.first, (int, np.integer))
            or isinstance(self.first, bool)
            or not isinstance(self.last, (int, np.integer))
            or isinstance(self.last, bool)
            or self.first <= 0
            or self.last < self.first
        ):
            raise ContractError("state range must satisfy 1 <= first <= last")
        object.__setattr__(self, "first", int(self.first))
        object.__setattr__(self, "last", int(self.last))

    @property
    def count(self) -> int:
        return self.last - self.first + 1


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Versioned numerical acceptance thresholds."""

    name: str
    raw_residual_tolerance_ev: float
    normalized_residual_tolerance: float
    s_orthogonality_tolerance: float

    @classmethod
    def for_precision(cls, precision: str) -> "QualityPolicy":
        if precision == "float32":
            return cls("quality-v1-float32", 1.0e-3, 1.0e-5, 1.0e-4)
        if precision == "float64":
            return cls("quality-v1-float64", 1.0e-8, 1.0e-10, 1.0e-8)
        raise ContractError("precision must be 'float32' or 'float64'")

    def __post_init__(self) -> None:
        if not self.name:
            raise ContractError("quality policy name must not be empty")
        for name, value in (
            ("raw_residual_tolerance_ev", self.raw_residual_tolerance_ev),
            (
                "normalized_residual_tolerance",
                self.normalized_residual_tolerance,
            ),
            (
                "s_orthogonality_tolerance",
                self.s_orthogonality_tolerance,
            ),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True, eq=False)
class NumericalQuality:
    """Generalized-eigenproblem quality retained with solved output."""

    raw_residuals_ev: np.ndarray
    normalized_residuals: np.ndarray
    s_orthogonality_error: float | None
    calculation_dtype: str
    policy: QualityPolicy

    def __post_init__(self) -> None:
        raw = frozen_array(
            self.raw_residuals_ev,
            name="raw_residuals_ev",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        normalized = frozen_array(
            self.normalized_residuals,
            name="normalized_residuals",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        if raw.shape != normalized.shape:
            raise ContractError("raw and normalized residual arrays must match")
        if np.any(raw < 0.0) or np.any(normalized < 0.0):
            raise ContractError("residuals must be nonnegative")
        orthogonality = self.s_orthogonality_error
        if orthogonality is not None and (
            not math.isfinite(orthogonality) or orthogonality < 0.0
        ):
            raise ContractError("S-orthogonality error must be nonnegative")
        if not self.calculation_dtype:
            raise ContractError("calculation_dtype must not be empty")
        if not isinstance(self.policy, QualityPolicy):
            raise ContractError("policy must be QualityPolicy")
        object.__setattr__(self, "raw_residuals_ev", raw)
        object.__setattr__(self, "normalized_residuals", normalized)

    @property
    def warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.raw_residuals_ev.size and float(
            np.max(self.raw_residuals_ev)
        ) > self.policy.raw_residual_tolerance_ev:
            warnings.append("raw generalized residual exceeds policy")
        if self.normalized_residuals.size and float(
            np.max(self.normalized_residuals)
        ) > self.policy.normalized_residual_tolerance:
            warnings.append("normalized generalized residual exceeds policy")
        if (
            self.s_orthogonality_error is not None
            and self.s_orthogonality_error
            > self.policy.s_orthogonality_tolerance
        ):
            warnings.append("S-orthogonality error exceeds policy")
        return tuple(warnings)

    @property
    def status(self) -> str:
        return "quality_warning" if self.warnings else "passed"


@dataclass(frozen=True, slots=True)
class SolverReceipt:
    """Resource and timing evidence for one solver call."""

    backend: str
    device: str
    scalar_dtype: str
    stage_seconds: Mapping[str, float]
    peak_host_rss_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    counters: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend or not self.device or not self.scalar_dtype:
            raise ContractError("solver receipt identity fields must not be empty")
        stages = {str(key): float(value) for key, value in self.stage_seconds.items()}
        if not stages or any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in stages.items()
        ):
            raise ContractError("solver receipt stage timings are invalid")
        counters = {str(key): int(value) for key, value in self.counters.items()}
        if any(not key or value < 0 for key, value in counters.items()):
            raise ContractError("solver receipt counters are invalid")
        for name, value in (
            ("peak_host_rss_bytes", self.peak_host_rss_bytes),
            ("peak_gpu_memory_bytes", self.peak_gpu_memory_bytes),
        ):
            if value is not None and (
                isinstance(value, bool) or int(value) < 0
            ):
                raise ContractError(f"{name} must be nonnegative or None")
        object.__setattr__(self, "stage_seconds", MappingProxyType(stages))
        object.__setattr__(self, "counters", MappingProxyType(counters))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

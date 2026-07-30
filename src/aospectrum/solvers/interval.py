"""Narrow energy-interval solver contract used by Band."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from aospectrum.assembly.pattern import SparseMatrixInput
from aospectrum.errors import ContractError
from aospectrum.model._arrays import frozen_array
from aospectrum.model.spectra import (
    EnergyInterval,
    NumericalQuality,
    QualityPolicy,
    SolverReceipt,
)


@dataclass(frozen=True, slots=True)
class IntervalSolveRequest:
    """Scientific request for all states in one open energy interval."""

    interval: EnergyInterval
    precision: str
    retain_vectors: bool = False
    quality_policy: QualityPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.interval, EnergyInterval):
            raise ContractError("interval must be EnergyInterval")
        if self.precision not in {"float32", "float64"}:
            raise ContractError("precision must be 'float32' or 'float64'")
        if type(self.retain_vectors) is not bool:
            raise ContractError("retain_vectors must be bool")
        policy = self.quality_policy or QualityPolicy.for_precision(self.precision)
        if not isinstance(policy, QualityPolicy):
            raise ContractError("quality_policy must be QualityPolicy")
        object.__setattr__(self, "quality_policy", policy)


@dataclass(frozen=True, slots=True, eq=False)
class IntervalSpectrum:
    """Complete interval spectrum with no fabricated absolute state numbers."""

    energies_ev: np.ndarray
    quality: NumericalQuality
    backend: str
    receipt: SolverReceipt
    complete: bool
    eigenvectors: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energies = frozen_array(
            self.energies_ev,
            name="interval energies",
            ndim=1,
            kinds="f",
        )
        if energies.size and np.any(np.diff(energies) < 0.0):
            raise ContractError("interval energies must be sorted")
        if not isinstance(self.quality, NumericalQuality):
            raise ContractError("quality must be NumericalQuality")
        if energies.size != self.quality.raw_residuals_ev.size:
            raise ContractError("energies and quality arrays must match")
        vectors = self.eigenvectors
        if vectors is not None:
            vectors = frozen_array(
                vectors,
                name="interval eigenvectors",
                ndim=2,
                kinds="fc",
            )
            if vectors.shape[1] != energies.size:
                raise ContractError("eigenvector columns must match energies")
        if not self.backend or not isinstance(self.receipt, SolverReceipt):
            raise ContractError("interval backend/receipt is invalid")
        if type(self.complete) is not bool:
            raise ContractError("complete must be bool")
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class EnergyIntervalSolver(Protocol):
    """Backend that solves exactly one explicit energy interval."""

    name: str

    def solve_interval(
        self,
        matrices: SparseMatrixInput,
        request: IntervalSolveRequest,
    ) -> IntervalSpectrum: ...


"""Narrow absolute-state solver contract used by Orbital."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from aospectrum.assembly.pattern import SparseMatrixInput
from aospectrum.errors import ContractError
from aospectrum.model._arrays import frozen_array
from aospectrum.model.spectra import (
    AbsoluteStateRange,
    NumericalQuality,
    QualityPolicy,
    SolverReceipt,
)


@dataclass(frozen=True, slots=True)
class IndexedSolveRequest:
    """Request a certified one-based inclusive state range."""

    selection: AbsoluteStateRange
    precision: str
    energy_hint_ev: float | None = None
    quality_policy: QualityPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, AbsoluteStateRange):
            raise ContractError("selection must be AbsoluteStateRange")
        if self.precision not in {"float32", "float64"}:
            raise ContractError("precision must be 'float32' or 'float64'")
        hint = self.energy_hint_ev
        if hint is not None and not math.isfinite(float(hint)):
            raise ContractError("energy_hint_ev must be finite when provided")
        policy = self.quality_policy or QualityPolicy.for_precision(self.precision)
        if not isinstance(policy, QualityPolicy):
            raise ContractError("quality_policy must be QualityPolicy")
        object.__setattr__(
            self,
            "energy_hint_ev",
            None if hint is None else float(hint),
        )
        object.__setattr__(self, "quality_policy", policy)


@dataclass(frozen=True, slots=True, eq=False)
class IndexedEigenpairs:
    """Certified absolute state numbers and selected eigenvectors."""

    absolute_state_numbers: np.ndarray
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    quality: NumericalQuality
    backend: str
    receipt: SolverReceipt
    locator_evidence: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = frozen_array(
            self.absolute_state_numbers,
            name="absolute_state_numbers",
            ndim=1,
            dtype=np.int64,
            kinds="iu",
        )
        energies = frozen_array(
            self.energies_ev,
            name="indexed energies",
            ndim=1,
            kinds="f",
        )
        vectors = frozen_array(
            self.eigenvectors,
            name="indexed eigenvectors",
            ndim=2,
            kinds="fc",
        )
        if states.size == 0 or states.shape != energies.shape:
            raise ContractError("indexed states and energies must match")
        if np.any(np.diff(states) != 1) or np.any(states <= 0):
            raise ContractError("absolute state numbers must be positive and contiguous")
        if energies.size and np.any(np.diff(energies) < 0.0):
            raise ContractError("indexed energies must be sorted")
        if vectors.shape[1] != states.size:
            raise ContractError("indexed eigenvector columns must match states")
        if not isinstance(self.quality, NumericalQuality):
            raise ContractError("quality must be NumericalQuality")
        if states.size != self.quality.raw_residuals_ev.size:
            raise ContractError("indexed states and quality arrays must match")
        if not self.backend or not isinstance(self.receipt, SolverReceipt):
            raise ContractError("indexed backend/receipt is invalid")
        if not self.locator_evidence:
            raise ContractError("indexed result requires locator evidence")
        object.__setattr__(self, "absolute_state_numbers", states)
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(
            self,
            "locator_evidence",
            MappingProxyType(dict(self.locator_evidence)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class IndexedStateSolver(Protocol):
    """Backend that returns exactly the certified requested state range."""

    name: str

    def solve_states(
        self,
        matrices: SparseMatrixInput,
        request: IndexedSolveRequest,
    ) -> IndexedEigenpairs: ...


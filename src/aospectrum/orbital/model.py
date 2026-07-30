"""Orbital requests, grid semantics and selected eigensystem results."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from aospectrum.errors import ContractError
from aospectrum.model._arrays import frozen_array
from aospectrum.model.spectra import (
    AbsoluteStateRange,
    NumericalQuality,
    SolverReceipt,
)
from aospectrum.orbital.volume import OrbitalField


@dataclass(frozen=True, slots=True)
class GridSpec:
    """One half-open periodic grid specified by shape or target spacing."""

    shape: tuple[int, int, int] | None = None
    spacing_angstrom: float | None = None

    def __post_init__(self) -> None:
        if (self.shape is None) == (self.spacing_angstrom is None):
            raise ContractError(
                "grid requires exactly one of shape or spacing_angstrom"
            )
        if self.shape is not None:
            values = tuple(self.shape)
            if len(values) != 3 or any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value <= 1
                for value in values
            ):
                raise ContractError(
                    "grid shape must contain three integers greater than one"
                )
            object.__setattr__(
                self,
                "shape",
                tuple(int(value) for value in values),
            )
        else:
            assert self.spacing_angstrom is not None
            spacing = float(self.spacing_angstrom)
            if not math.isfinite(spacing) or spacing <= 0.0:
                raise ContractError(
                    "grid spacing must be positive and finite"
                )
            object.__setattr__(self, "spacing_angstrom", spacing)

    def resolve_shape(
        self,
        cell_angstrom: np.ndarray,
    ) -> tuple[int, int, int]:
        if self.shape is not None:
            return self.shape
        assert self.spacing_angstrom is not None
        lengths = np.linalg.norm(
            np.asarray(cell_angstrom, dtype=np.float64),
            axis=1,
        )
        return tuple(
            max(2, int(math.ceil(length / self.spacing_angstrom)))
            for length in lengths
        )


@dataclass(frozen=True, slots=True)
class OrbitalRequest:
    """Single-frame Gamma-point Orbital request."""

    states: str
    precision: str = "float32"
    grid: GridSpec = field(default_factory=lambda: GridSpec(shape=(64, 64, 64)))
    energy_hint_ev: float | None = None
    degeneracy_notice_ev: float = 1.0e-5

    def __post_init__(self) -> None:
        if not isinstance(self.states, str) or not self.states.strip():
            raise ContractError("Orbital states selection must not be empty")
        if self.precision not in {"float32", "float64"}:
            raise ContractError("Orbital precision must be float32 or float64")
        if not isinstance(self.grid, GridSpec):
            raise ContractError("Orbital grid must be GridSpec")
        if self.energy_hint_ev is not None and not math.isfinite(
            float(self.energy_hint_ev)
        ):
            raise ContractError("Orbital energy hint must be finite")
        if (
            not math.isfinite(self.degeneracy_notice_ev)
            or self.degeneracy_notice_ev <= 0.0
        ):
            raise ContractError(
                "Orbital degeneracy notice threshold must be positive"
            )


@dataclass(frozen=True, slots=True)
class ResolvedStateSelection:
    """Original user expression resolved to one absolute state range."""

    expression: str
    absolute: AbsoluteStateRange
    semantics: str

    def __post_init__(self) -> None:
        if not self.expression or self.semantics not in {
            "absolute",
            "band_edge",
        }:
            raise ContractError("resolved state selection is invalid")


@dataclass(frozen=True, slots=True, eq=False)
class OrbitalEigensystem:
    """Phase-fixed selected states before real-space field evaluation."""

    selection: ResolvedStateSelection
    absolute_state_numbers: np.ndarray
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    phase_anchor_indices: np.ndarray
    quality: NumericalQuality
    receipt: SolverReceipt
    backend: str
    degeneracy_notices: tuple[str, ...] = ()
    locator_evidence: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        states = frozen_array(
            self.absolute_state_numbers,
            name="Orbital state numbers",
            ndim=1,
            dtype=np.int64,
            kinds="iu",
        )
        energies = frozen_array(
            self.energies_ev,
            name="Orbital energies",
            ndim=1,
            kinds="f",
        )
        vectors = frozen_array(
            self.eigenvectors,
            name="Orbital eigenvectors",
            ndim=2,
            kinds="fc",
        )
        anchors = frozen_array(
            self.phase_anchor_indices,
            name="Orbital phase anchors",
            ndim=1,
            dtype=np.int64,
            kinds="iu",
        )
        if (
            states.shape != energies.shape
            or states.shape != anchors.shape
            or vectors.shape[1] != states.size
        ):
            raise ContractError("Orbital eigensystem arrays do not align")
        if not np.array_equal(
            states,
            np.arange(
                self.selection.absolute.first,
                self.selection.absolute.last + 1,
                dtype=np.int64,
            ),
        ):
            raise ContractError("Orbital state numbers differ from selection")
        if np.any(anchors < 0) or np.any(anchors >= vectors.shape[0]):
            raise ContractError("Orbital phase anchor is out of range")
        if not self.backend or not isinstance(self.receipt, SolverReceipt):
            raise ContractError("Orbital backend/receipt is invalid")
        object.__setattr__(self, "absolute_state_numbers", states)
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "phase_anchor_indices", anchors)
        object.__setattr__(
            self,
            "degeneracy_notices",
            tuple(self.degeneracy_notices),
        )
        object.__setattr__(
            self,
            "locator_evidence",
            MappingProxyType(dict(self.locator_evidence)),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class OrbitalStateField:
    """One selected state paired with its real-space field."""

    state_number: int
    label: str
    energy_ev: float
    field: OrbitalField

    def __post_init__(self) -> None:
        if isinstance(self.state_number, bool) or int(self.state_number) <= 0:
            raise ContractError("Orbital field state number must be positive")
        if not self.label or not math.isfinite(self.energy_ev):
            raise ContractError("Orbital field label/energy is invalid")
        if not isinstance(self.field, OrbitalField):
            raise ContractError("Orbital state field payload is invalid")
        object.__setattr__(self, "state_number", int(self.state_number))

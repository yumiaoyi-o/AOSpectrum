"""Band-path requests and immutable result data."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from aospectrum.errors import ContractError
from aospectrum.model._arrays import frozen_array
from aospectrum.model.operators import LocalizedOperatorBundle
from aospectrum.model.spectra import (
    EnergyInterval,
    NumericalQuality,
    SolverReceipt,
)

BAND_CONTINUITY_GUARD_EV_V1 = 0.02


@dataclass(frozen=True, slots=True, eq=False)
class KPath:
    """Ordered fractional reciprocal coordinates and labelled path nodes."""

    fractional_points: np.ndarray
    node_indices: np.ndarray
    node_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        points = frozen_array(
            self.fractional_points,
            name="fractional k points",
            ndim=2,
            dtype=np.float64,
            kinds="f",
        )
        nodes = frozen_array(
            self.node_indices,
            name="k-path node indices",
            ndim=1,
            dtype=np.int64,
            kinds="iu",
        )
        labels = tuple(self.node_labels)
        if points.shape[1:] != (3,) or points.shape[0] < 2:
            raise ContractError("k path must contain at least two 3D points")
        if (
            nodes.size < 2
            or nodes[0] != 0
            or nodes[-1] != points.shape[0] - 1
            or np.any(np.diff(nodes) <= 0)
        ):
            raise ContractError(
                "k-path nodes must increase from first to last point"
            )
        if len(labels) != nodes.size or not all(
            isinstance(label, str) and label.strip() for label in labels
        ):
            raise ContractError("each k-path node requires a nonempty label")
        object.__setattr__(self, "fractional_points", points)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "node_labels", labels)

    @classmethod
    def interpolate(
        cls,
        nodes: Sequence[Sequence[float]],
        labels: Sequence[str],
        *,
        points_per_segment: int,
    ) -> "KPath":
        """Create equal-count segments, retaining each shared node once."""

        coordinates = np.asarray(nodes, dtype=np.float64)
        if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
            raise ContractError("k-path nodes must have shape (n_nodes, 3)")
        if coordinates.shape[0] < 2:
            raise ContractError("at least two k-path nodes are required")
        if (
            isinstance(points_per_segment, bool)
            or int(points_per_segment) <= 0
        ):
            raise ContractError("points_per_segment must be a positive integer")
        count = int(points_per_segment)
        parts: list[np.ndarray] = []
        node_indices = [0]
        for index in range(coordinates.shape[0] - 1):
            segment = np.linspace(
                coordinates[index],
                coordinates[index + 1],
                count + 1,
                endpoint=True,
                dtype=np.float64,
            )
            parts.append(segment if index == 0 else segment[1:])
            node_indices.append(node_indices[-1] + count)
        return cls(
            fractional_points=np.concatenate(parts, axis=0),
            node_indices=np.asarray(node_indices, dtype=np.int64),
            node_labels=tuple(labels),
        )

    @property
    def n_points(self) -> int:
        return int(self.fractional_points.shape[0])

    def distances(self, cell_angstrom: np.ndarray) -> np.ndarray:
        """Return cumulative reciprocal distance in inverse Angstrom."""

        cell = np.asarray(cell_angstrom, dtype=np.float64)
        if cell.shape != (3, 3):
            raise ContractError("cell must have shape (3, 3)")
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
        cartesian = self.fractional_points @ reciprocal
        increments = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        distances = np.empty(self.n_points, dtype=np.float64)
        distances[0] = 0.0
        distances[1:] = np.cumsum(increments)
        distances.setflags(write=False)
        return distances


@dataclass(frozen=True, slots=True)
class EnergyReference:
    """Energy zero used by a Band request and its rendered result."""

    mode: str = "absolute"
    value_ev: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"absolute", "fixed", "bundle_fermi"}:
            raise ContractError(
                "energy reference must be absolute, fixed or bundle_fermi"
            )
        if self.mode == "fixed":
            if self.value_ev is None or not math.isfinite(float(self.value_ev)):
                raise ContractError("fixed energy reference requires value_ev")
            object.__setattr__(self, "value_ev", float(self.value_ev))
        elif self.value_ev is not None:
            raise ContractError(
                "value_ev is only valid for a fixed energy reference"
            )

    def resolve(self, bundle: LocalizedOperatorBundle) -> float:
        if self.mode == "absolute":
            return 0.0
        if self.mode == "fixed":
            assert self.value_ev is not None
            return self.value_ev
        if bundle.filling is None or bundle.filling.fermi_energy_ev is None:
            raise ContractError(
                "bundle_fermi reference requires bundle filling.fermi_energy_ev"
            )
        return bundle.filling.fermi_energy_ev


@dataclass(frozen=True, slots=True)
class BandRequest:
    """Scientific Band request independent of execution resources."""

    kpath: KPath
    display_interval_ev: EnergyInterval
    energy_reference: EnergyReference = field(default_factory=EnergyReference)
    precision: str = "float32"

    def __post_init__(self) -> None:
        if not isinstance(self.kpath, KPath):
            raise ContractError("kpath must be KPath")
        if not isinstance(self.display_interval_ev, EnergyInterval):
            raise ContractError("display_interval_ev must be EnergyInterval")
        if not isinstance(self.energy_reference, EnergyReference):
            raise ContractError("energy_reference must be EnergyReference")
        if self.precision not in {"float32", "float64"}:
            raise ContractError("precision must be float32 or float64")


@dataclass(frozen=True, slots=True, eq=False)
class BandPoint:
    """One complete interval spectrum at one path point."""

    index: int
    k_fractional: np.ndarray
    absolute_energies_ev: np.ndarray
    quality: NumericalQuality
    receipt: SolverReceipt
    assembly_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or int(self.index) < 0:
            raise ContractError("band point index must be nonnegative")
        kpoint = frozen_array(
            self.k_fractional,
            name="band point coordinate",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        energies = frozen_array(
            self.absolute_energies_ev,
            name="band point energies",
            ndim=1,
            kinds="f",
        )
        if kpoint.shape != (3,):
            raise ContractError("band point coordinate must contain three values")
        if energies.size and np.any(np.diff(energies) < 0.0):
            raise ContractError("band point energies must be sorted")
        if energies.size != self.quality.raw_residuals_ev.size:
            raise ContractError("band point energies and quality sizes differ")
        if not isinstance(self.receipt, SolverReceipt):
            raise ContractError("band point receipt is invalid")
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "k_fractional", kpoint)
        object.__setattr__(self, "absolute_energies_ev", energies)
        object.__setattr__(
            self,
            "assembly_metadata",
            MappingProxyType(dict(self.assembly_metadata)),
        )


@dataclass(frozen=True, slots=True)
class BandResult:
    """Ordered Band result with a stable energy-reference record."""

    kpath: KPath
    display_interval_ev: EnergyInterval
    solve_interval_ev: EnergyInterval
    reference_energy_ev: float
    points: tuple[BandPoint, ...]
    precision: str
    backend: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) != self.kpath.n_points:
            raise ContractError("Band result must contain every k-path point")
        if tuple(point.index for point in points) != tuple(range(len(points))):
            raise ContractError("Band result points must be ordered and complete")
        for point, expected in zip(points, self.kpath.fractional_points):
            if not np.array_equal(point.k_fractional, expected):
                raise ContractError("Band point coordinate differs from k path")
        if not math.isfinite(self.reference_energy_ev):
            raise ContractError("reference_energy_ev must be finite")
        if (
            self.solve_interval_ev.lower_ev
            > self.display_interval_ev.lower_ev
            or self.solve_interval_ev.upper_ev
            < self.display_interval_ev.upper_ev
        ):
            raise ContractError("Band solve interval must cover display interval")
        if self.precision not in {"float32", "float64"}:
            raise ContractError("Band result precision is invalid")
        if not self.backend:
            raise ContractError("Band result backend must not be empty")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def quality_status(self) -> str:
        return (
            "quality_warning"
            if any(point.quality.warnings for point in self.points)
            else "passed"
        )

    @property
    def display_energies_ev(self) -> tuple[np.ndarray, ...]:
        return tuple(
            point.absolute_energies_ev - self.reference_energy_ev
            for point in self.points
        )

"""Complex orbital fields and probability-based isosurface levels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from aospectrum.errors import ContractError
from aospectrum.model._arrays import frozen_array


@dataclass(frozen=True, slots=True, eq=False)
class OrbitalField:
    """One complex field sampled on a half-open periodic grid."""

    values: np.ndarray
    grid_shape: tuple[int, int, int]
    cell_angstrom: np.ndarray
    origin_angstrom: np.ndarray
    voxel_volume_bohr3: float
    compute_device: str
    precision: str

    def __post_init__(self) -> None:
        expected = (
            np.dtype(np.complex64)
            if self.precision == "float32"
            else np.dtype(np.complex128)
        )
        if self.precision not in {"float32", "float64"}:
            raise ContractError("field precision must be float32 or float64")
        values = frozen_array(
            self.values,
            name="orbital field",
            ndim=3,
            kinds="c",
        )
        cell = frozen_array(
            self.cell_angstrom,
            name="field cell",
            ndim=2,
            dtype=np.float64,
            kinds="f",
        )
        origin = frozen_array(
            self.origin_angstrom,
            name="field origin",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        if values.dtype != expected or values.shape != self.grid_shape:
            raise ContractError(
                "orbital field dtype/shape differs from its provenance"
            )
        if cell.shape != (3, 3) or origin.shape != (3,):
            raise ContractError("orbital field geometry is invalid")
        if (
            not math.isfinite(self.voxel_volume_bohr3)
            or self.voxel_volume_bohr3 <= 0.0
        ):
            raise ContractError("field voxel volume must be positive")
        if not self.compute_device:
            raise ContractError("field compute device must not be empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "cell_angstrom", cell)
        object.__setattr__(self, "origin_angstrom", origin)

    @property
    def captured_norm(self) -> float:
        return float(
            np.sum(np.abs(self.values) ** 2, dtype=np.float64)
            * self.voxel_volume_bohr3
        )


@dataclass(frozen=True, slots=True, eq=False)
class EnclosedProbabilityLevels:
    """Absolute density/amplitude thresholds indexed by probability mass."""

    probabilities: np.ndarray
    density_levels: np.ndarray
    amplitude_levels: np.ndarray
    captured_norm: float

    def __post_init__(self) -> None:
        probabilities = frozen_array(
            self.probabilities,
            name="enclosed probabilities",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        density = frozen_array(
            self.density_levels,
            name="density isolevels",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        amplitude = frozen_array(
            self.amplitude_levels,
            name="amplitude isolevels",
            ndim=1,
            dtype=np.float64,
            kinds="f",
        )
        if (
            probabilities.shape != density.shape
            or probabilities.shape != amplitude.shape
            or probabilities.size < 2
            or np.any(np.diff(probabilities) <= 0.0)
            or probabilities[0] <= 0.0
            or probabilities[-1] >= 1.0
            or np.any(density < 0.0)
            or np.any(amplitude < 0.0)
        ):
            raise ContractError("enclosed-probability isolevel table is invalid")
        if not math.isfinite(self.captured_norm) or self.captured_norm <= 0.0:
            raise ContractError("captured orbital norm must be positive")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "density_levels", density)
        object.__setattr__(self, "amplitude_levels", amplitude)


def enclosed_probability_levels(
    field: OrbitalField,
    probabilities: np.ndarray | None = None,
) -> EnclosedProbabilityLevels:
    """Convert enclosed probability targets into absolute NGL isolevels."""

    requested = np.asarray(
        (
            np.linspace(0.50, 0.99, 50, dtype=np.float64)
            if probabilities is None
            else probabilities
        ),
        dtype=np.float64,
    )
    density = np.asarray(np.abs(field.values) ** 2, dtype=np.float64).reshape(-1)
    total = float(np.sum(density) * field.voxel_volume_bohr3)
    if not math.isfinite(total) or total <= 0.0:
        raise ContractError("orbital field has no positive captured norm")
    descending = np.sort(density)[::-1]
    cumulative = (
        np.cumsum(descending, dtype=np.float64)
        * field.voxel_volume_bohr3
        / total
    )
    positions = np.searchsorted(cumulative, requested, side="left")
    positions = np.clip(positions, 0, descending.size - 1)
    levels = descending[positions]
    return EnclosedProbabilityLevels(
        probabilities=requested,
        density_levels=levels,
        amplitude_levels=np.sqrt(levels),
        captured_norm=total,
    )

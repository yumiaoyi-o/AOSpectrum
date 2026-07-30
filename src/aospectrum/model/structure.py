"""Atomic geometry used by AO operator and field consumers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from aospectrum.errors import ContractError

from ._arrays import frozen_array


@dataclass(frozen=True, slots=True, eq=False)
class AtomicStructure:
    """One periodic structure in Angstrom."""

    cell_angstrom: np.ndarray
    positions_angstrom: np.ndarray
    atomic_numbers: np.ndarray
    periodic: tuple[bool, bool, bool] = (True, True, True)
    label: str | None = None

    def __post_init__(self) -> None:
        cell = frozen_array(
            self.cell_angstrom,
            name="cell_angstrom",
            ndim=2,
            dtype=np.float64,
            kinds="f",
        )
        positions = frozen_array(
            self.positions_angstrom,
            name="positions_angstrom",
            ndim=2,
            dtype=np.float64,
            kinds="f",
        )
        atomic_numbers = frozen_array(
            self.atomic_numbers,
            name="atomic_numbers",
            ndim=1,
            dtype=np.int32,
            kinds="iu",
        )
        if cell.shape != (3, 3):
            raise ContractError(
                f"cell_angstrom must have shape (3, 3), got {cell.shape}"
            )
        if not math.isfinite(float(np.linalg.det(cell))) or abs(
            float(np.linalg.det(cell))
        ) <= np.finfo(np.float64).eps:
            raise ContractError("cell_angstrom must be nonsingular")
        if positions.shape != (atomic_numbers.size, 3):
            raise ContractError(
                "positions_angstrom must have shape (n_atoms, 3)"
            )
        if atomic_numbers.size == 0 or np.any(atomic_numbers <= 0):
            raise ContractError("atomic_numbers must be nonempty and positive")
        periodic = tuple(self.periodic)
        if len(periodic) != 3 or not all(type(value) is bool for value in periodic):
            raise ContractError("periodic must contain exactly three booleans")
        label = self.label
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ContractError("structure label must be nonempty when provided")
        object.__setattr__(self, "cell_angstrom", cell)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "atomic_numbers", atomic_numbers)
        object.__setattr__(self, "periodic", periodic)

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.size)


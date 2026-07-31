"""Small data model shared by AOSpectrum Band and Orbital."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import InputError


def _array(
    value: Any,
    *,
    dtype: np.dtype[Any] | type | None = None,
    ndim: int | None = None,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise InputError(f"{name} must be {ndim}-dimensional")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class AtomicStructure:
    cell_angstrom: np.ndarray
    positions_angstrom: np.ndarray
    atomic_numbers: np.ndarray
    periodic: tuple[bool, bool, bool] = (True, True, True)
    label: str | None = None

    def __post_init__(self) -> None:
        cell = _array(
            self.cell_angstrom,
            dtype=np.float64,
            ndim=2,
            name="cell_angstrom",
        )
        positions = _array(
            self.positions_angstrom,
            dtype=np.float64,
            ndim=2,
            name="positions_angstrom",
        )
        numbers = _array(
            self.atomic_numbers,
            dtype=np.int32,
            ndim=1,
            name="atomic_numbers",
        )
        if cell.shape != (3, 3) or positions.shape != (numbers.size, 3):
            raise InputError("structure arrays have incompatible shapes")
        if numbers.size == 0:
            raise InputError("structure has no atoms")
        periodic = tuple(bool(value) for value in self.periodic)
        if len(periodic) != 3:
            raise InputError("periodic must contain three values")
        object.__setattr__(self, "cell_angstrom", cell)
        object.__setattr__(self, "positions_angstrom", positions)
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "periodic", periodic)

    @property
    def n_atoms(self) -> int:
        return int(self.atomic_numbers.size)


@dataclass(frozen=True, slots=True)
class OrbitalBasisLayout:
    n_atoms: int
    orbital_atom: np.ndarray
    local_index: np.ndarray
    basis_family: str
    basis_identity: str
    orbital_labels: tuple[str, ...] = ()
    angular_momentum: np.ndarray | None = None
    radial_index: np.ndarray | None = None
    real_harmonic_index: np.ndarray | None = None
    spinor_width: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    _atom_offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        atoms = _array(
            self.orbital_atom,
            dtype=np.int32,
            ndim=1,
            name="orbital_atom",
        )
        local = _array(
            self.local_index,
            dtype=np.int32,
            ndim=1,
            name="local_index",
        )
        n_atoms = int(self.n_atoms)
        if (
            n_atoms <= 0
            or atoms.size == 0
            or atoms.shape != local.shape
            or np.any(atoms < 0)
            or np.any(atoms >= n_atoms)
            or np.any(np.diff(atoms) < 0)
        ):
            raise InputError("invalid AO-to-atom layout")
        counts = np.bincount(atoms, minlength=n_atoms)
        if np.any(counts == 0):
            raise InputError("each atom must own at least one AO")
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        expected_local = np.arange(atoms.size, dtype=np.int64) - offsets[atoms]
        if not np.array_equal(local, expected_local):
            raise InputError("local AO indices must be contiguous within each atom")
        descriptors = (
            self.angular_momentum,
            self.radial_index,
            self.real_harmonic_index,
        )
        if any(value is not None for value in descriptors):
            if not all(value is not None for value in descriptors):
                raise InputError("orbital descriptors must be supplied together")
            angular = _array(
                self.angular_momentum,
                dtype=np.int32,
                ndim=1,
                name="angular_momentum",
            )
            radial = _array(
                self.radial_index,
                dtype=np.int32,
                ndim=1,
                name="radial_index",
            )
            harmonic = _array(
                self.real_harmonic_index,
                dtype=np.int32,
                ndim=1,
                name="real_harmonic_index",
            )
            if not (
                angular.shape == radial.shape == harmonic.shape == atoms.shape
            ):
                raise InputError("orbital descriptor lengths do not match AO layout")
        else:
            angular = radial = harmonic = None
        labels = tuple(str(value) for value in self.orbital_labels)
        if labels and len(labels) != atoms.size:
            raise InputError("orbital_labels must contain one value per AO")
        spinor_width = int(self.spinor_width)
        if spinor_width < 1:
            raise InputError("spinor_width must be positive")
        offsets.setflags(write=False)
        object.__setattr__(self, "n_atoms", n_atoms)
        object.__setattr__(self, "orbital_atom", atoms)
        object.__setattr__(self, "local_index", local)
        object.__setattr__(self, "orbital_labels", labels)
        object.__setattr__(self, "angular_momentum", angular)
        object.__setattr__(self, "radial_index", radial)
        object.__setattr__(self, "real_harmonic_index", harmonic)
        object.__setattr__(self, "spinor_width", spinor_width)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "_atom_offsets", offsets)

    @property
    def n_orbitals(self) -> int:
        return int(self.orbital_atom.size)

    @property
    def atom_offsets(self) -> np.ndarray:
        return self._atom_offsets

    @property
    def has_real_space_descriptors(self) -> bool:
        return self.angular_momentum is not None

    def orbital_count(self, atom: int) -> int:
        return int(self._atom_offsets[atom + 1] - self._atom_offsets[atom])


@dataclass(frozen=True, slots=True)
class ElectronicFilling:
    electron_count: int
    spin_degeneracy: int = 2
    fermi_energy_ev: float | None = None

    def __post_init__(self) -> None:
        electrons = int(self.electron_count)
        degeneracy = int(self.spin_degeneracy)
        if electrons <= 0 or degeneracy <= 0 or electrons % degeneracy:
            raise InputError("invalid closed-shell filling")
        object.__setattr__(self, "electron_count", electrons)
        object.__setattr__(self, "spin_degeneracy", degeneracy)
        if self.fermi_energy_ev is not None:
            object.__setattr__(
                self,
                "fermi_energy_ev",
                float(self.fermi_energy_ev),
            )

    @property
    def vbm_state_number(self) -> int:
        return self.electron_count // self.spin_degeneracy

    @property
    def cbm_state_number(self) -> int:
        return self.vbm_state_number + 1


@dataclass(frozen=True, slots=True)
class LocalizedOperatorBlocks:
    row_atom: np.ndarray
    col_atom: np.ndarray
    lattice_shift: np.ndarray
    block_offsets: np.ndarray
    block_shapes: np.ndarray
    h_values: np.ndarray
    s_values: np.ndarray

    def __post_init__(self) -> None:
        rows = _array(self.row_atom, dtype=np.int32, ndim=1, name="row_atom")
        cols = _array(self.col_atom, dtype=np.int32, ndim=1, name="col_atom")
        shifts = _array(
            self.lattice_shift,
            dtype=np.int32,
            ndim=2,
            name="lattice_shift",
        )
        offsets = _array(
            self.block_offsets,
            dtype=np.int64,
            ndim=1,
            name="block_offsets",
        )
        shapes = _array(
            self.block_shapes,
            dtype=np.int32,
            ndim=2,
            name="block_shapes",
        )
        h_values = _array(self.h_values, ndim=1, name="h_values")
        s_values = _array(self.s_values, ndim=1, name="s_values")
        n_blocks = rows.size
        if (
            n_blocks == 0
            or cols.shape != rows.shape
            or np.any(rows < 0)
            or np.any(cols < 0)
            or shifts.shape != (n_blocks, 3)
            or shapes.shape != (n_blocks, 2)
            or offsets.shape != (n_blocks + 1,)
            or offsets[0] != 0
        ):
            raise InputError("invalid operator block table")
        if not np.array_equal(np.diff(offsets), np.prod(shapes, axis=1)):
            raise InputError("operator block offsets do not match block shapes")
        if h_values.shape != s_values.shape or h_values.size != offsets[-1]:
            raise InputError("H/S values do not match the operator blocks")
        if h_values.dtype not in (np.float32, np.complex64):
            h_values = _array(
                h_values,
                dtype=np.complex64 if np.iscomplexobj(h_values) else np.float32,
                ndim=1,
                name="h_values",
            )
        if s_values.dtype not in (np.float32, np.complex64):
            s_values = _array(
                s_values,
                dtype=np.complex64 if np.iscomplexobj(s_values) else np.float32,
                ndim=1,
                name="s_values",
            )
        object.__setattr__(self, "row_atom", rows)
        object.__setattr__(self, "col_atom", cols)
        object.__setattr__(self, "lattice_shift", shifts)
        object.__setattr__(self, "block_offsets", offsets)
        object.__setattr__(self, "block_shapes", shapes)
        object.__setattr__(self, "h_values", h_values)
        object.__setattr__(self, "s_values", s_values)

    @property
    def n_blocks(self) -> int:
        return int(self.row_atom.size)


@dataclass(frozen=True, slots=True)
class LocalizedOperatorBundle:
    structure: AtomicStructure
    basis: OrbitalBasisLayout
    operators: LocalizedOperatorBlocks
    filling: ElectronicFilling | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.structure.n_atoms != self.basis.n_atoms:
            raise InputError("structure and basis atom counts differ")
        if np.any(self.operators.row_atom >= self.structure.n_atoms) or np.any(
            self.operators.col_atom >= self.structure.n_atoms
        ):
            raise InputError("operator block references an unknown atom")
        row_sizes = np.diff(self.basis.atom_offsets)[self.operators.row_atom]
        col_sizes = np.diff(self.basis.atom_offsets)[self.operators.col_atom]
        if not np.array_equal(
            self.operators.block_shapes,
            np.column_stack((row_sizes, col_sizes)),
        ):
            raise InputError("operator block shapes do not match the AO layout")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def n_orbitals(self) -> int:
        return self.basis.n_orbitals


@dataclass(frozen=True, slots=True)
class EnergyInterval:
    lower_ev: float
    upper_ev: float

    def __post_init__(self) -> None:
        lower = float(self.lower_ev)
        upper = float(self.upper_ev)
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise InputError("energy interval requires lower < upper")
        object.__setattr__(self, "lower_ev", lower)
        object.__setattr__(self, "upper_ev", upper)

    def expanded(self, guard_ev: float) -> "EnergyInterval":
        return EnergyInterval(
            self.lower_ev - float(guard_ev),
            self.upper_ev + float(guard_ev),
        )


@dataclass(frozen=True, slots=True)
class StateRange:
    first: int
    last: int

    def __post_init__(self) -> None:
        first = int(self.first)
        last = int(self.last)
        if first < 1 or last < first:
            raise InputError("state range requires 1 <= first <= last")
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "last", last)

    @property
    def count(self) -> int:
        return self.last - self.first + 1

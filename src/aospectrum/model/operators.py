"""Localized Hamiltonian/overlap blocks and electronic filling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from aospectrum.errors import ContractError

from ._arrays import frozen_array, scalar_family_bits
from .basis import OrbitalBasisLayout
from .structure import AtomicStructure


@dataclass(frozen=True, slots=True, eq=False)
class ElectronicFilling:
    """Electronic filling used only when VBM/CBM-relative semantics are requested."""

    mode: str
    electron_count: int
    spin_degeneracy: int = 2
    fermi_energy_ev: float | None = None

    def __post_init__(self) -> None:
        if self.mode != "closed_shell":
            raise ContractError("v1 ElectronicFilling mode must be 'closed_shell'")
        if (
            not isinstance(self.electron_count, (int, np.integer))
            or isinstance(self.electron_count, bool)
            or self.electron_count <= 0
        ):
            raise ContractError("electron_count must be a positive integer")
        if (
            not isinstance(self.spin_degeneracy, (int, np.integer))
            or isinstance(self.spin_degeneracy, bool)
            or self.spin_degeneracy <= 0
        ):
            raise ContractError("spin_degeneracy must be a positive integer")
        if int(self.electron_count) % int(self.spin_degeneracy) != 0:
            raise ContractError(
                "closed-shell electron_count must be divisible by spin_degeneracy"
            )
        fermi = self.fermi_energy_ev
        if fermi is not None and not math.isfinite(float(fermi)):
            raise ContractError("fermi_energy_ev must be finite when provided")
        object.__setattr__(self, "electron_count", int(self.electron_count))
        object.__setattr__(self, "spin_degeneracy", int(self.spin_degeneracy))
        object.__setattr__(
            self,
            "fermi_energy_ev",
            None if fermi is None else float(fermi),
        )

    @property
    def vbm_state_number(self) -> int:
        return self.electron_count // self.spin_degeneracy

    @property
    def cbm_state_number(self) -> int:
        return self.vbm_state_number + 1


@dataclass(frozen=True, slots=True, eq=False)
class LocalizedOperatorBlocks:
    """Ragged translation-indexed H(R)/S(R) block records in eV."""

    row_atom: np.ndarray
    col_atom: np.ndarray
    lattice_shift: np.ndarray
    block_offsets: np.ndarray
    block_shapes: np.ndarray
    h_values: np.ndarray
    s_values: np.ndarray

    def __post_init__(self) -> None:
        row_atom = frozen_array(
            self.row_atom,
            name="row_atom",
            ndim=1,
            dtype=np.int32,
            kinds="iu",
        )
        col_atom = frozen_array(
            self.col_atom,
            name="col_atom",
            ndim=1,
            dtype=np.int32,
            kinds="iu",
        )
        lattice_shift = frozen_array(
            self.lattice_shift,
            name="lattice_shift",
            ndim=2,
            dtype=np.int32,
            kinds="iu",
        )
        block_offsets = frozen_array(
            self.block_offsets,
            name="block_offsets",
            ndim=1,
            dtype=np.int64,
            kinds="iu",
        )
        block_shapes = frozen_array(
            self.block_shapes,
            name="block_shapes",
            ndim=2,
            dtype=np.int32,
            kinds="iu",
        )
        h_values = frozen_array(
            self.h_values,
            name="h_values",
            ndim=1,
            kinds="fc",
        )
        s_values = frozen_array(
            self.s_values,
            name="s_values",
            ndim=1,
            kinds="fc",
        )
        n_blocks = int(row_atom.size)
        if n_blocks == 0 or col_atom.shape != row_atom.shape:
            raise ContractError("row_atom/col_atom must be equal nonempty arrays")
        if lattice_shift.shape != (n_blocks, 3):
            raise ContractError("lattice_shift must have shape (n_blocks, 3)")
        if block_shapes.shape != (n_blocks, 2) or np.any(block_shapes <= 0):
            raise ContractError("block_shapes must have shape (n_blocks, 2)")
        if block_offsets.shape != (n_blocks + 1,):
            raise ContractError("block_offsets must have length n_blocks + 1")
        if block_offsets[0] != 0 or np.any(np.diff(block_offsets) <= 0):
            raise ContractError("block_offsets must be strictly increasing from zero")
        expected_sizes = np.prod(block_shapes, axis=1, dtype=np.int64)
        if not np.array_equal(np.diff(block_offsets), expected_sizes):
            raise ContractError("block offsets do not match ragged block shapes")
        if h_values.shape != s_values.shape or h_values.size != block_offsets[-1]:
            raise ContractError("H/S values do not match the ragged block table")
        if scalar_family_bits(h_values.dtype) != scalar_family_bits(s_values.dtype):
            raise ContractError("H/S values must use the same scalar precision")
        object.__setattr__(self, "row_atom", row_atom)
        object.__setattr__(self, "col_atom", col_atom)
        object.__setattr__(self, "lattice_shift", lattice_shift)
        object.__setattr__(self, "block_offsets", block_offsets)
        object.__setattr__(self, "block_shapes", block_shapes)
        object.__setattr__(self, "h_values", h_values)
        object.__setattr__(self, "s_values", s_values)

    @property
    def n_blocks(self) -> int:
        return int(self.row_atom.size)

    @property
    def precision_bits(self) -> int:
        return scalar_family_bits(self.h_values.dtype)

    def values_for_block(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(index, bool) or not 0 <= int(index) < self.n_blocks:
            raise ContractError("block index is out of range")
        start = int(self.block_offsets[int(index)])
        stop = int(self.block_offsets[int(index) + 1])
        shape = tuple(int(value) for value in self.block_shapes[int(index)])
        return (
            self.h_values[start:stop].reshape(shape),
            self.s_values[start:stop].reshape(shape),
        )


@dataclass(frozen=True, slots=True, eq=False)
class LocalizedOperatorBundle:
    """Complete, producer-neutral input consumed by Band and Orbital."""

    structure: AtomicStructure
    basis: OrbitalBasisLayout
    operators: LocalizedOperatorBlocks
    filling: ElectronicFilling | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "aospectrum.operator-bundle/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.structure, AtomicStructure):
            raise ContractError("structure must be AtomicStructure")
        if not isinstance(self.basis, OrbitalBasisLayout):
            raise ContractError("basis must be OrbitalBasisLayout")
        if not isinstance(self.operators, LocalizedOperatorBlocks):
            raise ContractError("operators must be LocalizedOperatorBlocks")
        if self.filling is not None and not isinstance(
            self.filling, ElectronicFilling
        ):
            raise ContractError("filling must be ElectronicFilling or None")
        if self.schema != "aospectrum.operator-bundle/v1":
            raise ContractError("unsupported operator bundle schema")
        if self.basis.n_atoms != self.structure.n_atoms:
            raise ContractError("basis and structure atom counts differ")
        if np.any(self.operators.row_atom < 0) or np.any(
            self.operators.row_atom >= self.structure.n_atoms
        ):
            raise ContractError("row_atom contains an out-of-range atom")
        if np.any(self.operators.col_atom < 0) or np.any(
            self.operators.col_atom >= self.structure.n_atoms
        ):
            raise ContractError("col_atom contains an out-of-range atom")
        row_counts = np.asarray(
            [
                self.basis.orbital_count(int(atom))
                for atom in self.operators.row_atom
            ],
            dtype=np.int32,
        )
        col_counts = np.asarray(
            [
                self.basis.orbital_count(int(atom))
                for atom in self.operators.col_atom
            ],
            dtype=np.int32,
        )
        if not np.array_equal(
            self.operators.block_shapes,
            np.column_stack((row_counts, col_counts)),
        ):
            raise ContractError(
                "operator block shapes do not match the AO basis layout"
            )
        if (
            self.filling is not None
            and self.filling.vbm_state_number >= self.basis.n_orbitals
        ):
            raise ContractError("electronic filling exceeds the AO state space")
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )

    @property
    def n_orbitals(self) -> int:
        return self.basis.n_orbitals

    @property
    def precision_bits(self) -> int:
        return self.operators.precision_bits

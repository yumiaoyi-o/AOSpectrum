"""Reusable complex64/float32 sparse Bloch assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse

from .data import LocalizedOperatorBundle
from .errors import InputError


_INT32_MAX = np.iinfo(np.int32).max


def _indices(values: np.ndarray) -> np.ndarray:
    dtype = np.int32 if int(np.max(values, initial=0)) <= _INT32_MAX else np.int64
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _project_hermitian(
    values: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    diagonal: np.ndarray,
) -> None:
    if upper.size:
        mean = (values[upper] + np.conjugate(values[lower])) * 0.5
        values[upper] = mean
        values[lower] = np.conjugate(mean)
    if np.iscomplexobj(values) and diagonal.size:
        values[diagonal] = np.real(values[diagonal])


@dataclass(slots=True)
class SparsePair:
    hamiltonian: sparse.csr_matrix
    overlap: sparse.csr_matrix
    k_fractional: tuple[float, float, float]

    @property
    def n_orbitals(self) -> int:
        return int(self.hamiltonian.shape[0])


@dataclass(frozen=True, slots=True)
class BlochPattern:
    n_orbitals: int
    csr_indices: np.ndarray
    csr_indptr: np.ndarray
    source_indices: np.ndarray
    phase_groups: np.ndarray
    group_starts: np.ndarray
    phase_shifts: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    diagonal: np.ndarray

    @classmethod
    def from_bundle(cls, bundle: LocalizedOperatorBundle) -> "BlochPattern":
        operators = bundle.operators
        offsets = bundle.basis.atom_offsets
        shifts, block_groups = np.unique(
            operators.lattice_shift,
            axis=0,
            return_inverse=True,
        )
        rows_parts: list[np.ndarray] = []
        columns_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        group_parts: list[np.ndarray] = []
        for block in range(operators.n_blocks):
            row_atom = int(operators.row_atom[block])
            col_atom = int(operators.col_atom[block])
            n_rows, n_columns = (
                int(value) for value in operators.block_shapes[block]
            )
            rows_parts.append(
                int(offsets[row_atom])
                + np.repeat(np.arange(n_rows, dtype=np.int64), n_columns)
            )
            columns_parts.append(
                int(offsets[col_atom])
                + np.tile(np.arange(n_columns, dtype=np.int64), n_rows)
            )
            source = np.arange(
                int(operators.block_offsets[block]),
                int(operators.block_offsets[block + 1]),
                dtype=np.int64,
            )
            source_parts.append(source)
            group_parts.append(
                np.full(source.size, block_groups[block], dtype=np.int64)
            )

        rows = np.concatenate(rows_parts)
        columns = np.concatenate(columns_parts)
        sources = np.concatenate(source_parts)
        groups = np.concatenate(group_parts)
        keys = rows * bundle.n_orbitals + columns
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        first = np.empty(sorted_keys.size, dtype=bool)
        first[0] = True
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        starts = np.flatnonzero(first)
        unique = sorted_keys[starts]
        unique_rows = unique // bundle.n_orbitals
        unique_columns = unique % bundle.n_orbitals

        mirror_keys = unique_columns * bundle.n_orbitals + unique_rows
        mirror = np.searchsorted(unique, mirror_keys)
        if np.any(mirror >= unique.size) or not np.array_equal(
            unique[mirror],
            mirror_keys,
        ):
            raise InputError("operator topology is not Hermitian")
        upper = np.flatnonzero(unique_rows < unique_columns)
        row_counts = np.bincount(unique_rows, minlength=bundle.n_orbitals)
        indptr = np.empty(bundle.n_orbitals + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        phase_shifts = np.asarray(shifts, dtype=np.int32)
        phase_shifts.setflags(write=False)
        return cls(
            n_orbitals=bundle.n_orbitals,
            csr_indices=_indices(unique_columns),
            csr_indptr=_indices(indptr),
            source_indices=_indices(sources[order]),
            phase_groups=_indices(groups[order]),
            group_starts=_indices(starts),
            phase_shifts=phase_shifts,
            upper=_indices(upper),
            lower=_indices(mirror[upper]),
            diagonal=_indices(np.flatnonzero(unique_rows == unique_columns)),
        )

    @property
    def nnz(self) -> int:
        return int(self.csr_indices.size)

    def workspace(self, *, complex_values: bool) -> "BlochWorkspace":
        dtype = np.complex64 if complex_values else np.float32
        shape = (self.n_orbitals, self.n_orbitals)
        hamiltonian = sparse.csr_matrix(
            (
                np.empty(self.nnz, dtype=dtype),
                self.csr_indices,
                self.csr_indptr,
            ),
            shape=shape,
            copy=False,
        )
        overlap = sparse.csr_matrix(
            (
                np.empty(self.nnz, dtype=dtype),
                self.csr_indices,
                self.csr_indptr,
            ),
            shape=shape,
            copy=False,
        )
        pair = SparsePair(hamiltonian, overlap, (0.0, 0.0, 0.0))
        return BlochWorkspace(
            pattern=self,
            pair=pair,
            scratch=np.empty(self.source_indices.size, dtype=dtype),
        )


@dataclass(slots=True)
class BlochWorkspace:
    pattern: BlochPattern
    pair: SparsePair
    scratch: np.ndarray

    def _fill(
        self,
        source: np.ndarray,
        phases: np.ndarray,
        output: np.ndarray,
    ) -> None:
        self.scratch[...] = source[self.pattern.source_indices]
        np.multiply(
            self.scratch,
            phases[self.pattern.phase_groups],
            out=self.scratch,
        )
        np.add.reduceat(
            self.scratch,
            self.pattern.group_starts,
            out=output,
        )

    def update(
        self,
        bundle: LocalizedOperatorBundle,
        k_fractional: Iterable[float],
    ) -> SparsePair:
        kpoint = tuple(float(value) for value in k_fractional)
        if len(kpoint) != 3:
            raise InputError("k point must contain three values")
        phases = np.exp(
            np.asarray(2.0j * np.pi, dtype=np.complex64)
            * (
                self.pattern.phase_shifts.astype(np.float32)
                @ np.asarray(kpoint, dtype=np.float32)
            )
        )
        if self.scratch.dtype == np.float32:
            if np.iscomplexobj(bundle.operators.h_values) or np.iscomplexobj(
                bundle.operators.s_values
            ):
                raise InputError("Gamma Orbital requires real H/S")
            phases = np.real(phases).astype(np.float32)
        else:
            phases = phases.astype(np.complex64)
        self._fill(
            bundle.operators.h_values,
            phases,
            self.pair.hamiltonian.data,
        )
        self._fill(
            bundle.operators.s_values,
            phases,
            self.pair.overlap.data,
        )
        _project_hermitian(
            self.pair.hamiltonian.data,
            self.pattern.upper,
            self.pattern.lower,
            self.pattern.diagonal,
        )
        _project_hermitian(
            self.pair.overlap.data,
            self.pattern.upper,
            self.pattern.lower,
            self.pattern.diagonal,
        )
        self.pair.k_fractional = kpoint
        return self.pair


def assemble_bloch(
    bundle: LocalizedOperatorBundle,
    k_fractional: Iterable[float],
    *,
    complex_values: bool = True,
) -> SparsePair:
    pattern = BlochPattern.from_bundle(bundle)
    return pattern.workspace(complex_values=complex_values).update(
        bundle,
        k_fractional,
    )

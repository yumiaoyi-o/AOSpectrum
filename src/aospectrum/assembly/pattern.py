"""Reusable sparse Bloch assembly from ragged AO block records."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
from scipy import sparse

from aospectrum.errors import AssemblyError
from aospectrum.model.operators import LocalizedOperatorBundle


HERMITIAN_CLOSURE_EPSILON_MULTIPLIER = 64.0
_INT32_MAX = np.iinfo(np.int32).max
_SUPPORTED_DTYPES = {
    np.dtype(np.float32),
    np.dtype(np.complex64),
    np.dtype(np.float64),
    np.dtype(np.complex128),
}


def _compact_indices(values: np.ndarray) -> np.ndarray:
    dtype = np.int32 if not values.size or int(np.max(values)) <= _INT32_MAX else np.int64
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def _readonly(values: np.ndarray) -> np.ndarray:
    values.setflags(write=False)
    return values


def _max_hermiticity_error(
    values: np.ndarray,
    *,
    upper_positions: np.ndarray,
    lower_positions: np.ndarray,
    diagonal_positions: np.ndarray,
) -> float:
    pair_error = 0.0
    if upper_positions.size:
        pair_error = float(
            np.max(
                np.abs(
                    values[upper_positions]
                    - np.conjugate(values[lower_positions])
                )
            )
        )
    diagonal_error = 0.0
    if np.iscomplexobj(values) and diagonal_positions.size:
        diagonal_error = float(
            np.max(2.0 * np.abs(np.imag(values[diagonal_positions])))
        )
    return max(pair_error, diagonal_error)


def _bounded_hermitian_values(
    values: np.ndarray,
    *,
    name: str,
    source_dtype: np.dtype,
    upper_positions: np.ndarray,
    lower_positions: np.ndarray,
    diagonal_positions: np.ndarray,
) -> tuple[float, float, float, bool]:
    error_before = _max_hermiticity_error(
        values,
        upper_positions=upper_positions,
        lower_positions=lower_positions,
        diagonal_positions=diagonal_positions,
    )
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    real_dtype = np.empty((), dtype=source_dtype).real.dtype
    tolerance = (
        HERMITIAN_CLOSURE_EPSILON_MULTIPLIER
        * float(np.finfo(real_dtype).eps)
        * scale
    )
    if error_before > tolerance:
        raise AssemblyError(
            f"{name}(k) exceeds roundoff_projection_v1: "
            f"max_skew={error_before:.6e}, allowed={tolerance:.6e}"
        )
    applied = error_before != 0.0
    if applied:
        if upper_positions.size:
            averaged = (
                values[upper_positions]
                + np.conjugate(values[lower_positions])
            ) * 0.5
            values[upper_positions] = averaged
            values[lower_positions] = np.conjugate(averaged)
        if np.iscomplexobj(values) and diagonal_positions.size:
            values[diagonal_positions] = np.real(values[diagonal_positions])
    error_after = _max_hermiticity_error(
        values,
        upper_positions=upper_positions,
        lower_positions=lower_positions,
        diagonal_positions=diagonal_positions,
    )
    if error_after != 0.0:
        raise AssemblyError(
            f"{name}(k) roundoff projection was not exactly Hermitian"
        )
    return error_before, tolerance, error_after, applied


@dataclass(frozen=True, slots=True)
class PreparedBlochPhase:
    """One k-point phase vector in the requested scalar family."""

    k_fractional: tuple[float, float, float]
    values: np.ndarray
    dtype: np.dtype


@dataclass(frozen=True, slots=True)
class SparseMatrixPair:
    """Owned immutable H(k)/S(k) pair in eV."""

    hamiltonian: sparse.csr_matrix
    overlap: sparse.csr_matrix
    k_fractional: tuple[float, float, float]
    h_hermiticity_error: float
    s_hermiticity_error: float
    h_error_before_projection: float = 0.0
    s_error_before_projection: float = 0.0
    h_projection_tolerance: float = 0.0
    s_projection_tolerance: float = 0.0
    h_projection_applied: bool = False
    s_projection_applied: bool = False

    @property
    def n_orbitals(self) -> int:
        return int(self.hamiltonian.shape[0])


@dataclass(slots=True)
class BorrowedSparsePair:
    """Workspace-owned matrices valid until the next numeric update."""

    hamiltonian: sparse.csr_matrix
    overlap: sparse.csr_matrix
    k_fractional: tuple[float, float, float]
    h_hermiticity_error: float = 0.0
    s_hermiticity_error: float = 0.0
    h_error_before_projection: float = 0.0
    s_error_before_projection: float = 0.0
    h_projection_tolerance: float = 0.0
    s_projection_tolerance: float = 0.0
    h_projection_applied: bool = False
    s_projection_applied: bool = False
    generation: int = 0

    @property
    def n_orbitals(self) -> int:
        return int(self.hamiltonian.shape[0])

    def freeze(self) -> SparseMatrixPair:
        hamiltonian = self.hamiltonian.copy()
        overlap = self.overlap.copy()
        for matrix in (hamiltonian, overlap):
            matrix.data.setflags(write=False)
            matrix.indices.setflags(write=False)
            matrix.indptr.setflags(write=False)
        return SparseMatrixPair(
            hamiltonian=hamiltonian,
            overlap=overlap,
            k_fractional=self.k_fractional,
            h_hermiticity_error=self.h_hermiticity_error,
            s_hermiticity_error=self.s_hermiticity_error,
            h_error_before_projection=self.h_error_before_projection,
            s_error_before_projection=self.s_error_before_projection,
            h_projection_tolerance=self.h_projection_tolerance,
            s_projection_tolerance=self.s_projection_tolerance,
            h_projection_applied=self.h_projection_applied,
            s_projection_applied=self.s_projection_applied,
        )


SparseMatrixInput = SparseMatrixPair | BorrowedSparsePair


@dataclass(frozen=True, slots=True, eq=False)
class BlochPattern:
    """Immutable CSR structure and source contribution map."""

    n_orbitals: int
    basis_identity: str
    atom_offsets: np.ndarray
    row_atom: np.ndarray
    col_atom: np.ndarray
    lattice_shift: np.ndarray
    block_offsets: np.ndarray
    block_shapes: np.ndarray
    csr_indices: np.ndarray
    csr_indptr: np.ndarray
    ordered_source_indices: np.ndarray
    ordered_phase_groups: np.ndarray
    group_starts: np.ndarray
    phase_shifts: np.ndarray
    upper_positions: np.ndarray
    lower_positions: np.ndarray
    diagonal_positions: np.ndarray

    @classmethod
    def from_bundle(cls, bundle: LocalizedOperatorBundle) -> "BlochPattern":
        operators = bundle.operators
        offsets = bundle.basis.atom_offsets
        unique_shifts, block_phase_groups = np.unique(
            operators.lattice_shift,
            axis=0,
            return_inverse=True,
        )
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []
        phase_parts: list[np.ndarray] = []
        for block in range(operators.n_blocks):
            row_atom = int(operators.row_atom[block])
            col_atom = int(operators.col_atom[block])
            n_rows, n_cols = (
                int(value) for value in operators.block_shapes[block]
            )
            rows = int(offsets[row_atom]) + np.repeat(
                np.arange(n_rows, dtype=np.int64),
                n_cols,
            )
            cols = int(offsets[col_atom]) + np.tile(
                np.arange(n_cols, dtype=np.int64),
                n_rows,
            )
            source = np.arange(
                int(operators.block_offsets[block]),
                int(operators.block_offsets[block + 1]),
                dtype=np.int64,
            )
            rows_parts.append(rows)
            cols_parts.append(cols)
            source_parts.append(source)
            phase_parts.append(
                np.full(source.size, block_phase_groups[block], dtype=np.int64)
            )
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
        source_indices = np.concatenate(source_parts)
        phase_groups = np.concatenate(phase_parts)
        coordinate_keys = rows * bundle.n_orbitals + cols
        order = np.argsort(coordinate_keys, kind="stable")
        sorted_keys = coordinate_keys[order]
        first = np.empty(sorted_keys.size, dtype=bool)
        first[0] = True
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        group_starts = np.flatnonzero(first)
        unique_keys = sorted_keys[group_starts]
        unique_rows = unique_keys // bundle.n_orbitals
        unique_cols = unique_keys % bundle.n_orbitals

        mirror_keys = unique_cols * bundle.n_orbitals + unique_rows
        mirror_positions = np.searchsorted(unique_keys, mirror_keys)
        valid = mirror_positions < unique_keys.size
        if not np.all(valid) or not np.array_equal(
            unique_keys[mirror_positions],
            mirror_keys,
        ):
            raise AssemblyError(
                "operator block topology is not structurally Hermitian"
            )
        upper = np.flatnonzero(unique_rows < unique_cols)
        lower = mirror_positions[upper]
        diagonal = np.flatnonzero(unique_rows == unique_cols)
        row_counts = np.bincount(unique_rows, minlength=bundle.n_orbitals)
        indptr = np.empty(bundle.n_orbitals + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(row_counts, out=indptr[1:])
        return cls(
            n_orbitals=bundle.n_orbitals,
            basis_identity=bundle.basis.basis_identity,
            atom_offsets=bundle.basis.atom_offsets,
            row_atom=operators.row_atom,
            col_atom=operators.col_atom,
            lattice_shift=operators.lattice_shift,
            block_offsets=operators.block_offsets,
            block_shapes=operators.block_shapes,
            csr_indices=_compact_indices(unique_cols),
            csr_indptr=_compact_indices(indptr),
            ordered_source_indices=_compact_indices(source_indices[order]),
            ordered_phase_groups=_compact_indices(phase_groups[order]),
            group_starts=_compact_indices(group_starts),
            phase_shifts=_readonly(np.asarray(unique_shifts, dtype=np.int32)),
            upper_positions=_compact_indices(upper),
            lower_positions=_compact_indices(lower),
            diagonal_positions=_compact_indices(diagonal),
        )

    @property
    def nnz(self) -> int:
        return int(self.csr_indices.size)

    @property
    def contribution_count(self) -> int:
        return int(self.ordered_source_indices.size)

    @property
    def structural_nbytes(self) -> int:
        return sum(
            int(value.nbytes)
            for value in (
                self.csr_indices,
                self.csr_indptr,
                self.ordered_source_indices,
                self.ordered_phase_groups,
                self.group_starts,
                self.phase_shifts,
                self.upper_positions,
                self.lower_positions,
                self.diagonal_positions,
            )
        )

    def _require_compatible(self, bundle: LocalizedOperatorBundle) -> None:
        operators = bundle.operators
        if (
            bundle.n_orbitals != self.n_orbitals
            or bundle.basis.basis_identity != self.basis_identity
            or not np.array_equal(bundle.basis.atom_offsets, self.atom_offsets)
            or not np.array_equal(operators.row_atom, self.row_atom)
            or not np.array_equal(operators.col_atom, self.col_atom)
            or not np.array_equal(operators.lattice_shift, self.lattice_shift)
            or not np.array_equal(operators.block_offsets, self.block_offsets)
            or not np.array_equal(operators.block_shapes, self.block_shapes)
        ):
            raise AssemblyError("operator bundle topology does not match pattern")

    def prepare_phase(
        self,
        bundle: LocalizedOperatorBundle,
        k_fractional: Iterable[float],
        *,
        precision: str,
        force_complex: bool = False,
    ) -> PreparedBlochPhase:
        self._require_compatible(bundle)
        if precision not in {"float32", "float64"}:
            raise AssemblyError("precision must be 'float32' or 'float64'")
        kpoint = tuple(float(value) for value in k_fractional)
        if len(kpoint) != 3 or not all(math.isfinite(value) for value in kpoint):
            raise AssemblyError("k_fractional must contain three finite values")
        phases = np.exp(
            2.0j
            * np.pi
            * (np.asarray(self.phase_shifts, dtype=np.float64) @ np.asarray(kpoint))
        )
        real_dtype = np.dtype(np.float32 if precision == "float32" else np.float64)
        complex_dtype = np.dtype(
            np.complex64 if precision == "float32" else np.complex128
        )
        if type(force_complex) is not bool:
            raise AssemblyError("force_complex must be bool")
        needs_complex = force_complex or (
            np.iscomplexobj(bundle.operators.h_values)
            or np.iscomplexobj(bundle.operators.s_values)
            or bool(np.any(np.abs(np.imag(phases)) > 1.0e-14))
        )
        dtype = complex_dtype if needs_complex else real_dtype
        values = (
            phases.astype(complex_dtype)
            if needs_complex
            else np.real(phases).astype(real_dtype)
        )
        values.setflags(write=False)
        return PreparedBlochPhase(kpoint, values, dtype)

    def create_workspace(self, dtype: np.dtype | str) -> "BlochWorkspace":
        return BlochWorkspace.create(self, dtype=dtype)

    def assemble(
        self,
        bundle: LocalizedOperatorBundle,
        k_fractional: Iterable[float],
        *,
        precision: str,
    ) -> SparseMatrixPair:
        phase = self.prepare_phase(bundle, k_fractional, precision=precision)
        workspace = self.create_workspace(phase.dtype)
        return workspace.update(bundle, phase).freeze()


@dataclass(slots=True)
class BlochWorkspace:
    """Stable CSR buffers reused across k points with one topology."""

    pattern: BlochPattern
    dtype: np.dtype
    hamiltonian: sparse.csr_matrix
    overlap: sparse.csr_matrix
    scratch: np.ndarray = field(repr=False)
    view: BorrowedSparsePair = field(repr=False)

    @classmethod
    def create(
        cls,
        pattern: BlochPattern,
        *,
        dtype: np.dtype | str,
    ) -> "BlochWorkspace":
        scalar_dtype = np.dtype(dtype)
        if scalar_dtype not in _SUPPORTED_DTYPES:
            raise AssemblyError("unsupported sparse workspace dtype")
        shape = (pattern.n_orbitals, pattern.n_orbitals)
        hamiltonian = sparse.csr_matrix(
            (
                np.empty(pattern.nnz, dtype=scalar_dtype),
                pattern.csr_indices,
                pattern.csr_indptr,
            ),
            shape=shape,
            copy=False,
        )
        overlap = sparse.csr_matrix(
            (
                np.empty(pattern.nnz, dtype=scalar_dtype),
                pattern.csr_indices,
                pattern.csr_indptr,
            ),
            shape=shape,
            copy=False,
        )
        view = BorrowedSparsePair(
            hamiltonian=hamiltonian,
            overlap=overlap,
            k_fractional=(0.0, 0.0, 0.0),
        )
        return cls(
            pattern=pattern,
            dtype=scalar_dtype,
            hamiltonian=hamiltonian,
            overlap=overlap,
            scratch=np.empty(pattern.contribution_count, dtype=scalar_dtype),
            view=view,
        )

    @property
    def value_nbytes(self) -> int:
        return int(
            self.hamiltonian.data.nbytes
            + self.overlap.data.nbytes
            + self.scratch.nbytes
        )

    def _fill(
        self,
        source: np.ndarray,
        phase: PreparedBlochPhase,
        output: np.ndarray,
    ) -> None:
        self.scratch[...] = np.take(
            source,
            self.pattern.ordered_source_indices,
        )
        np.multiply(
            self.scratch,
            phase.values[self.pattern.ordered_phase_groups],
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
        phase: PreparedBlochPhase,
    ) -> BorrowedSparsePair:
        self.pattern._require_compatible(bundle)
        if phase.dtype != self.dtype:
            raise AssemblyError("phase dtype does not match workspace dtype")
        self._fill(
            bundle.operators.h_values,
            phase,
            self.hamiltonian.data,
        )
        (
            h_before,
            h_tolerance,
            h_after,
            h_applied,
        ) = _bounded_hermitian_values(
            self.hamiltonian.data,
            name="H",
            source_dtype=bundle.operators.h_values.dtype,
            upper_positions=self.pattern.upper_positions,
            lower_positions=self.pattern.lower_positions,
            diagonal_positions=self.pattern.diagonal_positions,
        )
        self._fill(
            bundle.operators.s_values,
            phase,
            self.overlap.data,
        )
        (
            s_before,
            s_tolerance,
            s_after,
            s_applied,
        ) = _bounded_hermitian_values(
            self.overlap.data,
            name="S",
            source_dtype=bundle.operators.s_values.dtype,
            upper_positions=self.pattern.upper_positions,
            lower_positions=self.pattern.lower_positions,
            diagonal_positions=self.pattern.diagonal_positions,
        )
        self.view.k_fractional = phase.k_fractional
        self.view.h_hermiticity_error = h_after
        self.view.s_hermiticity_error = s_after
        self.view.h_error_before_projection = h_before
        self.view.s_error_before_projection = s_before
        self.view.h_projection_tolerance = h_tolerance
        self.view.s_projection_tolerance = s_tolerance
        self.view.h_projection_applied = h_applied
        self.view.s_projection_applied = s_applied
        self.view.generation += 1
        return self.view


def assemble_bloch(
    bundle: LocalizedOperatorBundle,
    k_fractional: Iterable[float],
    *,
    precision: str = "float32",
) -> SparseMatrixPair:
    """Assemble one owned sparse H(k)/S(k) pair."""

    return BlochPattern.from_bundle(bundle).assemble(
        bundle,
        k_fractional,
        precision=precision,
    )

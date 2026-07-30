"""Quality diagnostics shared by interval and indexed-state solvers."""

from __future__ import annotations

import numpy as np

from aospectrum.assembly.pattern import SparseMatrixInput
from aospectrum.errors import SolverError
from aospectrum.model.spectra import NumericalQuality, QualityPolicy


def generalized_residual_metrics(
    matrices: SparseMatrixInput,
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw residuals in eV and scale-normalized residuals."""

    energies = np.asarray(energies_ev)
    vectors = np.asarray(eigenvectors)
    if energies.ndim != 1 or vectors.shape != (
        matrices.n_orbitals,
        energies.size,
    ):
        raise SolverError("eigenvalue/eigenvector shapes do not match H/S")
    raw = np.empty(energies.size, dtype=np.float64)
    normalized = np.empty(energies.size, dtype=np.float64)
    for index, energy in enumerate(energies):
        vector = vectors[:, index]
        h_vector = matrices.hamiltonian @ vector
        s_vector = matrices.overlap @ vector
        numerator = np.linalg.norm(h_vector - energy * s_vector)
        denominator = (
            np.linalg.norm(h_vector)
            + abs(energy) * np.linalg.norm(s_vector)
        )
        raw[index] = float(numerator)
        normalized[index] = (
            float(numerator / denominator)
            if denominator > 0.0
            else float(numerator)
        )
    return raw, normalized


def s_orthogonality_error(
    matrices: SparseMatrixInput,
    eigenvectors: np.ndarray,
) -> float:
    vectors = np.asarray(eigenvectors)
    if vectors.ndim != 2 or vectors.shape[0] != matrices.n_orbitals:
        raise SolverError("eigenvectors do not match overlap dimension")
    if vectors.shape[1] == 0:
        return 0.0
    gram = vectors.conj().T @ (matrices.overlap @ vectors)
    return float(
        np.max(np.abs(gram - np.eye(gram.shape[0], dtype=gram.dtype)))
    )


def assess_quality(
    matrices: SparseMatrixInput,
    energies_ev: np.ndarray,
    eigenvectors: np.ndarray,
    policy: QualityPolicy,
) -> NumericalQuality:
    raw, normalized = generalized_residual_metrics(
        matrices,
        energies_ev,
        eigenvectors,
    )
    dtype = np.result_type(
        matrices.hamiltonian.dtype,
        matrices.overlap.dtype,
        np.asarray(eigenvectors).dtype,
    ).name
    return NumericalQuality(
        raw_residuals_ev=raw,
        normalized_residuals=normalized,
        s_orthogonality_error=s_orthogonality_error(matrices, eigenvectors),
        calculation_dtype=dtype,
        policy=policy,
    )


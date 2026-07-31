"""Single-GPU complex64 CISS interval solver backed by cuDSS."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any

import numpy as np
from scipy import linalg, sparse

from .data import EnergyInterval
from .errors import InputError, SolverError
from .sparse import SparsePair


@dataclass(slots=True)
class CissResult:
    energies_ev: np.ndarray
    maximum_residual_ev: float
    stage_seconds: dict[str, float]
    peak_gpu_memory_bytes: int | None
    warnings: tuple[str, ...]
    complete: bool


@dataclass(slots=True)
class ShiftedResult:
    solutions: np.ndarray
    stage_seconds: dict[str, float]
    peak_gpu_memory_bytes: int | None


@dataclass(frozen=True, slots=True)
class CissSettings:
    integration_points: int = 32
    contour_batch_size: int = 1
    block_size: int = 32
    moment_size: int = 4
    ellipse_vertical_scale: float = 0.1
    endpoint_guard_ev: float = 1.0e-5
    rank_relative_tolerance: float = 1.0e-5
    spurious_relative_threshold: float = 1.0e-4
    random_seed: int = 20260727

    def __post_init__(self) -> None:
        if (
            self.integration_points < 1
            or self.contour_batch_size < 1
            or self.integration_points % self.contour_batch_size
            or self.block_size < 1
            or self.moment_size < 1
        ):
            raise InputError(
                "CISS sizes must be positive and contour_batch_size "
                "must divide integration_points"
            )

    @property
    def search_dimension(self) -> int:
        return self.block_size * self.moment_size


def _require_shared_pattern(matrices: SparsePair) -> None:
    hamiltonian = matrices.hamiltonian
    overlap = matrices.overlap
    if not sparse.isspmatrix_csr(hamiltonian) or not sparse.isspmatrix_csr(
        overlap
    ):
        raise InputError("CISS requires CSR H and S")
    if not np.array_equal(hamiltonian.indptr, overlap.indptr) or not np.array_equal(
        hamiltonian.indices,
        overlap.indices,
    ):
        raise InputError("CISS requires one shared H/S CSR pattern")
    if hamiltonian.dtype != np.complex64 or overlap.dtype != np.complex64:
        raise InputError("CISS requires complex64 H/S")


def _contour_geometry(
    interval: EnergyInterval,
    settings: CissSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = interval.lower_ev - 2.0 * settings.endpoint_guard_ev
    upper = interval.upper_ev + 2.0 * settings.endpoint_guard_ev
    center = 0.5 * (lower + upper)
    horizontal_radius = 0.5 * (upper - lower)
    vertical_radius = horizontal_radius * settings.ellipse_vertical_scale
    indices = np.arange(settings.integration_points, dtype=np.float32)
    theta = 2.0 * np.pi * (indices + 0.5) / settings.integration_points
    normalized_nodes = np.cos(theta) + (
        1j * settings.ellipse_vertical_scale * np.sin(theta)
    )
    shifts = center + horizontal_radius * normalized_nodes
    weights = (
        vertical_radius * np.cos(theta)
        + 1j * horizontal_radius * np.sin(theta)
    ) / settings.integration_points
    return (
        np.asarray(shifts, dtype=np.complex64),
        np.asarray(normalized_nodes, dtype=np.complex64),
        np.asarray(weights, dtype=np.complex64),
    )


def _compress_moments(
    matrices: SparsePair,
    moments: np.ndarray,
    relative_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    gram = moments.conj().T @ (matrices.overlap @ moments)
    gram = 0.5 * (gram + gram.conj().T)
    values, vectors = np.linalg.eigh(gram)
    scale = float(np.max(np.abs(values), initial=0.0))
    if scale == 0.0 or not math.isfinite(scale):
        raise SolverError("CISS contour moments have zero numerical rank")
    retained = values > scale * relative_tolerance
    if not np.any(retained):
        raise SolverError("CISS contour moments have no positive stable subspace")
    positive = values[retained]
    transform = vectors[:, retained] / np.sqrt(positive)[None, :]
    return (
        np.asarray(moments @ transform, dtype=np.complex64),
        np.sqrt(positive),
    )


def _rayleigh_ritz(
    matrices: SparsePair,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reduced_h = basis.conj().T @ (matrices.hamiltonian @ basis)
    reduced_s = basis.conj().T @ (matrices.overlap @ basis)
    reduced_h = 0.5 * (reduced_h + reduced_h.conj().T)
    reduced_s = 0.5 * (reduced_s + reduced_s.conj().T)
    try:
        energies, coefficients = linalg.eigh(
            reduced_h,
            reduced_s,
            check_finite=False,
            driver="gvd",
        )
    except linalg.LinAlgError as exc:
        raise SolverError("CISS reduced overlap is not positive definite") from exc
    return (
        np.asarray(energies, dtype=np.float32),
        np.asarray(basis @ coefficients, dtype=np.complex64),
        np.asarray(coefficients, dtype=np.complex64),
    )


def _nonspurious_mask(
    coefficients: np.ndarray,
    singular_values: np.ndarray,
    threshold: float,
) -> np.ndarray:
    amplitudes = np.abs(coefficients) ** 2
    numerators = np.sum(amplitudes, axis=0)
    denominators = np.sum(
        amplitudes / singular_values[:, None],
        axis=0,
    )
    scores = np.divide(
        numerators,
        denominators,
        out=np.zeros(numerators.shape, dtype=numerators.dtype),
        where=denominators > 0.0,
    )
    maximum = float(np.max(scores, initial=0.0))
    if maximum == 0.0 or not math.isfinite(maximum):
        raise SolverError("CISS ghost filter produced no positive finite score")
    retained = np.logical_and(
        np.isfinite(scores),
        scores >= threshold * maximum,
    )
    return retained


class NativeCudssShiftSolver:
    def __init__(self) -> None:
        self._session: Any | None = None
        self._pattern_token: tuple[Any, ...] | None = None

    @staticmethod
    def _token(matrices: SparsePair) -> tuple[Any, ...]:
        return (
            matrices.hamiltonian.shape,
            int(matrices.hamiltonian.indptr.__array_interface__["data"][0]),
            int(matrices.hamiltonian.indices.__array_interface__["data"][0]),
            matrices.hamiltonian.indptr.size,
            matrices.hamiltonian.indices.size,
        )

    def solve_shifted(
        self,
        matrices: SparsePair,
        shifts_ev: np.ndarray,
        rhs: np.ndarray,
    ) -> ShiftedResult:
        try:
            from aospectrum import _aospectrum_cuda
        except (ImportError, OSError) as exc:
            raise SolverError(
                "ciss-cudss requires the optional AOSpectrum CUDA extension"
            ) from exc
        token = self._token(matrices)
        if self._session is not None and token != self._pattern_token:
            raise SolverError("cuDSS session cannot change its CSR pattern")
        try:
            if self._session is None:
                self._session = _aospectrum_cuda.CudssShiftSession(
                    matrices.hamiltonian.indptr,
                    matrices.hamiltonian.indices,
                    matrices.hamiltonian.shape[0],
                    "complex64",
                )
                self._pattern_token = token
            started = perf_counter()
            native = self._session.solve(
                matrices.hamiltonian.data,
                matrices.overlap.data,
                shifts_ev,
                rhs,
            )
        except RuntimeError as exc:
            raise SolverError(f"cuDSS shifted solve failed: {exc}") from exc
        elapsed = perf_counter() - started
        return ShiftedResult(
            solutions=np.asarray(native.solutions),
            stage_seconds={
                "native_total": elapsed,
                **dict(native.stage_seconds),
            },
            peak_gpu_memory_bytes=getattr(
                native,
                "peak_gpu_memory_bytes",
                None,
            ),
        )

    def close(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except RuntimeError as exc:
                raise SolverError(f"cannot close cuDSS session: {exc}") from exc
            finally:
                self._session = None
                self._pattern_token = None


@dataclass(slots=True)
class CudssCissSolver:
    settings: CissSettings = field(default_factory=CissSettings)
    shifted_solver: NativeCudssShiftSolver | None = None

    def __post_init__(self) -> None:
        if self.shifted_solver is None:
            self.shifted_solver = NativeCudssShiftSolver()

    def close(self) -> None:
        assert self.shifted_solver is not None
        self.shifted_solver.close()

    def solve_interval(
        self,
        matrices: SparsePair,
        interval: EnergyInterval,
    ) -> CissResult:
        _require_shared_pattern(matrices)
        if self.settings.search_dimension >= matrices.n_orbitals:
            raise SolverError(
                "CISS search dimension must be smaller than matrix dimension"
            )
        shifts, normalized_nodes, weights = _contour_geometry(
            interval,
            self.settings,
        )
        stages: dict[str, float] = {}
        total_started = perf_counter()

        setup_started = perf_counter()
        rng = np.random.default_rng(self.settings.random_seed)
        shape = (matrices.n_orbitals, self.settings.block_size)
        probes = np.asarray(
            (
                rng.standard_normal(shape, dtype=np.float32)
                + 1j * rng.standard_normal(shape, dtype=np.float32)
            )
            / np.asarray(math.sqrt(2.0), dtype=np.float32),
            dtype=np.complex64,
        )
        probes, _ = np.linalg.qr(probes, mode="reduced")
        rhs = np.asarray(matrices.overlap @ probes, dtype=np.complex64)
        moments = np.zeros(
            (matrices.n_orbitals, self.settings.search_dimension),
            dtype=np.complex64,
        )
        stages["probe_setup"] = perf_counter() - setup_started

        native_peak_gpu: int | None = None
        batch_size = self.settings.contour_batch_size
        assert self.shifted_solver is not None
        for start in range(0, shifts.size, batch_size):
            stop = start + batch_size
            solve_started = perf_counter()
            result = self.shifted_solver.solve_shifted(
                matrices,
                shifts[start:stop],
                rhs,
            )
            stages["shifted_solves"] = stages.get("shifted_solves", 0.0) + (
                perf_counter() - solve_started
            )
            for name, seconds in result.stage_seconds.items():
                key = f"native_{name}"
                stages[key] = stages.get(key, 0.0) + float(seconds)
            if result.peak_gpu_memory_bytes is not None:
                native_peak_gpu = max(
                    native_peak_gpu or 0,
                    result.peak_gpu_memory_bytes,
                )
            for local in range(stop - start):
                solution = result.solutions[local]
                for moment in range(self.settings.moment_size):
                    first = moment * self.settings.block_size
                    last = first + self.settings.block_size
                    moments[:, first:last] += (
                        -weights[start + local]
                        * normalized_nodes[start + local] ** moment
                        * solution
                    )

        compression_started = perf_counter()
        basis, singular_values = _compress_moments(
            matrices,
            moments,
            self.settings.rank_relative_tolerance,
        )
        stages["subspace_compression"] = perf_counter() - compression_started
        subspace_saturated = basis.shape[1] == self.settings.search_dimension

        ritz_started = perf_counter()
        energies, vectors, coefficients = _rayleigh_ritz(
            matrices,
            basis,
        )
        stages["rayleigh_ritz"] = perf_counter() - ritz_started
        nonspurious = _nonspurious_mask(
            coefficients,
            singular_values,
            self.settings.spurious_relative_threshold,
        )
        inside = np.logical_and(
            energies > interval.lower_ev,
            energies < interval.upper_ev,
        )
        selected = np.logical_and(nonspurious, inside)
        selected_energies = energies[selected]
        selected_vectors = vectors[:, selected]
        if selected_energies.size:
            residuals = (
                matrices.hamiltonian @ selected_vectors
                - (matrices.overlap @ selected_vectors)
                * selected_energies[None, :]
            )
            maximum_residual = float(
                np.max(np.linalg.norm(residuals, axis=0))
            )
        else:
            maximum_residual = 0.0
        warnings: list[str] = []
        if maximum_residual > 1.0e-3:
            warnings.append(
                f"maximum eigenpair residual is {maximum_residual:.3e} eV"
            )
        if subspace_saturated:
            warnings.append(
                "CISS search subspace is saturated; the returned interval "
                "spectrum may be incomplete"
            )
        stages["solver_total"] = perf_counter() - total_started
        return CissResult(
            energies_ev=selected_energies,
            maximum_residual_ev=maximum_residual,
            stage_seconds=stages,
            peak_gpu_memory_bytes=native_peak_gpu,
            warnings=tuple(warnings),
            complete=not subspace_saturated,
        )

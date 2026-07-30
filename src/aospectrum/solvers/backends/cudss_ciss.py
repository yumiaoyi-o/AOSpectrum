"""Single-GPU CISS interval solver backed by cuDSS shifted solves."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from scipy import linalg, sparse

from aospectrum.assembly.pattern import SparseMatrixInput
from aospectrum.errors import ContractError, SolverError
from aospectrum.model.spectra import SolverReceipt
from aospectrum.solvers.interval import (
    IntervalSolveRequest,
    IntervalSpectrum,
)
from aospectrum.solvers.quality import assess_quality


def _receipt_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): _receipt_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_receipt_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ShiftedSolveReceipt:
    """Execution evidence returned by one shifted-system batch."""

    stage_seconds: Mapping[str, float]
    peak_gpu_memory_bytes: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stages = {str(key): float(value) for key, value in self.stage_seconds.items()}
        if not stages or any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in stages.items()
        ):
            raise ContractError("shifted-solve stage timings are invalid")
        if self.peak_gpu_memory_bytes is not None and (
            isinstance(self.peak_gpu_memory_bytes, bool)
            or int(self.peak_gpu_memory_bytes) < 0
        ):
            raise ContractError("peak GPU memory must be nonnegative or None")
        object.__setattr__(self, "stage_seconds", MappingProxyType(stages))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True, eq=False)
class ShiftedSolveResult:
    """Solutions for ``(H - zS) X = B`` at one shift batch."""

    solutions: np.ndarray
    receipt: ShiftedSolveReceipt

    def __post_init__(self) -> None:
        values = np.asarray(self.solutions)
        if values.ndim != 3 or values.dtype.kind != "c":
            raise ContractError(
                "shifted solutions must have shape (n_shifts, n, n_rhs)"
            )
        if not np.all(np.isfinite(values)):
            raise ContractError("shifted solutions must be finite")
        if values.flags.writeable:
            values = np.array(values, copy=True, order="C")
        values.setflags(write=False)
        if not isinstance(self.receipt, ShiftedSolveReceipt):
            raise ContractError("shifted result receipt is invalid")
        object.__setattr__(self, "solutions", values)


@runtime_checkable
class ShiftedBatchSolver(Protocol):
    """Minimal linear algebra boundary owned by the CISS implementation."""

    scalar_dtype: str

    def solve_shifted(
        self,
        matrices: SparseMatrixInput,
        shifts_ev: np.ndarray,
        rhs: np.ndarray,
    ) -> ShiftedSolveResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CissSettings:
    """Versioned internal CISS policy, not ordinary user configuration."""

    integration_points: int = 32
    contour_batch_size: int = 1
    block_size: int = 32
    moment_size: int = 4
    ellipse_vertical_scale: float = 0.1
    endpoint_guard_ev: float = 1.0e-5
    rank_relative_tolerance: float | None = None
    spurious_relative_threshold: float = 1.0e-4
    linear_residual_limit: float | None = None
    random_seed: int = 20260727

    def __post_init__(self) -> None:
        for name in (
            "integration_points",
            "contour_batch_size",
            "block_size",
            "moment_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ContractError(f"CISS {name} must be a positive integer")
        if self.integration_points % self.contour_batch_size != 0:
            raise ContractError(
                "integration_points must be divisible by contour_batch_size"
            )
        for name in (
            "ellipse_vertical_scale",
            "endpoint_guard_ev",
            "spurious_relative_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"CISS {name} must be positive and finite")
        if self.ellipse_vertical_scale > 1.0:
            raise ContractError("ellipse_vertical_scale must not exceed one")
        for name in ("rank_relative_tolerance", "linear_residual_limit"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise ContractError(f"CISS {name} must be positive when provided")
        if isinstance(self.random_seed, bool) or int(self.random_seed) < 0:
            raise ContractError("random_seed must be nonnegative")

    @property
    def search_dimension(self) -> int:
        return self.block_size * self.moment_size

    def rank_tolerance(self, precision: str) -> float:
        if self.rank_relative_tolerance is not None:
            return float(self.rank_relative_tolerance)
        return 1.0e-5 if precision == "float32" else 1.0e-10

    def linear_tolerance(self, precision: str) -> float:
        if self.linear_residual_limit is not None:
            return float(self.linear_residual_limit)
        return 5.0e-5 if precision == "float32" else 1.0e-10


def _scalar_dtypes(precision: str) -> tuple[np.dtype, np.dtype]:
    if precision == "float32":
        return np.dtype(np.float32), np.dtype(np.complex64)
    if precision == "float64":
        return np.dtype(np.float64), np.dtype(np.complex128)
    raise ContractError("precision must be 'float32' or 'float64'")


def _require_shared_pattern(
    matrices: SparseMatrixInput,
    precision: str,
) -> None:
    real_dtype, complex_dtype = _scalar_dtypes(precision)
    hamiltonian = matrices.hamiltonian
    overlap = matrices.overlap
    if not sparse.isspmatrix_csr(hamiltonian) or not sparse.isspmatrix_csr(
        overlap
    ):
        raise ContractError("CISS requires CSR H and S")
    if (
        not hamiltonian.has_canonical_format
        or not overlap.has_canonical_format
        or not hamiltonian.has_sorted_indices
        or not overlap.has_sorted_indices
    ):
        raise ContractError("CISS requires canonical sorted CSR matrices")
    if not np.array_equal(hamiltonian.indptr, overlap.indptr) or not np.array_equal(
        hamiltonian.indices,
        overlap.indices,
    ):
        raise ContractError("CISS requires H and S to share one CSR pattern")
    allowed = {real_dtype, complex_dtype}
    if hamiltonian.dtype not in allowed or overlap.dtype not in allowed:
        raise ContractError(
            f"CISS matrix dtype does not match {precision}: "
            f"H={hamiltonian.dtype}, S={overlap.dtype}"
        )


def _contour_geometry(
    request: IntervalSolveRequest,
    settings: CissSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real_dtype, complex_dtype = _scalar_dtypes(request.precision)
    lower = request.interval.lower_ev - 2.0 * settings.endpoint_guard_ev
    upper = request.interval.upper_ev + 2.0 * settings.endpoint_guard_ev
    center = 0.5 * (lower + upper)
    horizontal_radius = 0.5 * (upper - lower)
    vertical_radius = horizontal_radius * settings.ellipse_vertical_scale
    indices = np.arange(settings.integration_points, dtype=real_dtype)
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
        np.asarray(shifts, dtype=complex_dtype),
        np.asarray(normalized_nodes, dtype=complex_dtype),
        np.asarray(weights, dtype=complex_dtype),
    )


def _compress_moments(
    matrices: SparseMatrixInput,
    moments: np.ndarray,
    relative_tolerance: float,
    complex_dtype: np.dtype,
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
        np.asarray(moments @ transform, dtype=complex_dtype),
        np.sqrt(positive),
    )


def _rayleigh_ritz(
    matrices: SparseMatrixInput,
    basis: np.ndarray,
    real_dtype: np.dtype,
    complex_dtype: np.dtype,
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
        np.asarray(energies, dtype=real_dtype),
        np.asarray(basis @ coefficients, dtype=complex_dtype),
        np.asarray(coefficients, dtype=complex_dtype),
    )


def _nonspurious_mask(
    coefficients: np.ndarray,
    singular_values: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
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
    return retained, scores


class NativeCudssShiftSolver:
    """Typed session adapter for the optional ``_aospectrum_cuda`` extension."""

    def __init__(self, precision: str) -> None:
        if precision != "float32":
            raise SolverError("ciss-cudss supports complex64 only")
        _, complex_dtype = _scalar_dtypes(precision)
        self.scalar_dtype = complex_dtype.name
        self._session: Any | None = None
        self._pattern_token: tuple[Any, ...] | None = None

    @staticmethod
    def _token(matrices: SparseMatrixInput) -> tuple[Any, ...]:
        return (
            matrices.hamiltonian.shape,
            int(matrices.hamiltonian.indptr.__array_interface__["data"][0]),
            int(matrices.hamiltonian.indices.__array_interface__["data"][0]),
            matrices.hamiltonian.indptr.size,
            matrices.hamiltonian.indices.size,
        )

    def solve_shifted(
        self,
        matrices: SparseMatrixInput,
        shifts_ev: np.ndarray,
        rhs: np.ndarray,
    ) -> ShiftedSolveResult:
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
                    self.scalar_dtype,
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
        return ShiftedSolveResult(
            solutions=np.asarray(native.solutions),
            receipt=ShiftedSolveReceipt(
                stage_seconds={
                    "native_total": elapsed,
                    **dict(native.stage_seconds),
                },
                peak_gpu_memory_bytes=getattr(
                    native,
                    "peak_gpu_memory_bytes",
                    None,
                ),
                details=dict(getattr(native, "details", {})),
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
    """CISS contour solver whose heavy shifted systems are owned by cuDSS."""

    precision: str = "float32"
    settings: CissSettings = field(default_factory=CissSettings)
    shifted_solver: ShiftedBatchSolver | None = None
    name: str = field(init=False)

    def __post_init__(self) -> None:
        if self.precision != "float32":
            raise SolverError("ciss-cudss supports complex64 only")
        _, complex_dtype = _scalar_dtypes(self.precision)
        self.name = "ciss-cudss"
        if self.shifted_solver is None:
            self.shifted_solver = NativeCudssShiftSolver(self.precision)
        if self.shifted_solver.scalar_dtype != complex_dtype.name:
            raise ContractError(
                "CISS and shifted solver scalar dtypes do not match"
            )

    def close(self) -> None:
        assert self.shifted_solver is not None
        self.shifted_solver.close()

    def solve_interval(
        self,
        matrices: SparseMatrixInput,
        request: IntervalSolveRequest,
    ) -> IntervalSpectrum:
        if request.precision != self.precision:
            raise SolverError("CISS request precision differs from backend")
        _require_shared_pattern(matrices, request.precision)
        if self.settings.search_dimension >= matrices.n_orbitals:
            raise SolverError(
                "CISS search dimension must be smaller than matrix dimension"
            )
        real_dtype, complex_dtype = _scalar_dtypes(request.precision)
        shifts, normalized_nodes, weights = _contour_geometry(
            request,
            self.settings,
        )
        stages: dict[str, float] = {}
        total_started = perf_counter()

        setup_started = perf_counter()
        rng = np.random.default_rng(self.settings.random_seed)
        shape = (matrices.n_orbitals, self.settings.block_size)
        probes = np.asarray(
            (
                rng.standard_normal(shape, dtype=real_dtype)
                + 1j * rng.standard_normal(shape, dtype=real_dtype)
            )
            / np.asarray(math.sqrt(2.0), dtype=real_dtype),
            dtype=complex_dtype,
        )
        probes, _ = np.linalg.qr(probes, mode="reduced")
        rhs = np.asarray(matrices.overlap @ probes, dtype=complex_dtype)
        moments = np.zeros(
            (matrices.n_orbitals, self.settings.search_dimension),
            dtype=complex_dtype,
        )
        stages["probe_setup"] = perf_counter() - setup_started

        native_stage_totals: dict[str, float] = {}
        native_batch_details: list[dict[str, Any]] = []
        native_peak_gpu: int | None = None
        maximum_linear_residual = 0.0
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
            for name, seconds in result.receipt.stage_seconds.items():
                native_stage_totals[name] = (
                    native_stage_totals.get(name, 0.0) + float(seconds)
                )
            native_batch_details.append(
                _receipt_value(dict(result.receipt.details))
            )
            if result.receipt.peak_gpu_memory_bytes is not None:
                native_peak_gpu = max(
                    native_peak_gpu or 0,
                    result.receipt.peak_gpu_memory_bytes,
                )
            for local, shift in enumerate(shifts[start:stop]):
                solution = result.solutions[local]
                residual = (
                    matrices.hamiltonian @ solution
                    - shift * (matrices.overlap @ solution)
                    - rhs
                )
                denominators = np.linalg.norm(rhs, axis=0)
                relative = np.divide(
                    np.linalg.norm(residual, axis=0),
                    denominators,
                    out=np.full(denominators.shape, np.inf),
                    where=denominators > 0.0,
                )
                maximum_linear_residual = max(
                    maximum_linear_residual,
                    float(np.max(relative)),
                )
                for moment in range(self.settings.moment_size):
                    first = moment * self.settings.block_size
                    last = first + self.settings.block_size
                    moments[:, first:last] += (
                        -weights[start + local]
                        * normalized_nodes[start + local] ** moment
                        * solution
                    )
        if not math.isfinite(maximum_linear_residual):
            raise SolverError(
                "shifted-system residual is not finite"
            )
        linear_residual_warning = (
            maximum_linear_residual
            > self.settings.linear_tolerance(request.precision)
        )

        compression_started = perf_counter()
        basis, singular_values = _compress_moments(
            matrices,
            moments,
            self.settings.rank_tolerance(request.precision),
            complex_dtype,
        )
        stages["subspace_compression"] = perf_counter() - compression_started
        subspace_saturated = basis.shape[1] == self.settings.search_dimension

        ritz_started = perf_counter()
        energies, vectors, coefficients = _rayleigh_ritz(
            matrices,
            basis,
            real_dtype,
            complex_dtype,
        )
        stages["rayleigh_ritz"] = perf_counter() - ritz_started
        nonspurious, scores = _nonspurious_mask(
            coefficients,
            singular_values,
            self.settings.spurious_relative_threshold,
        )
        inside = np.logical_and(
            energies > request.interval.lower_ev,
            energies < request.interval.upper_ev,
        )
        selected = np.logical_and(nonspurious, inside)
        selected_energies = energies[selected]
        selected_vectors = vectors[:, selected]
        candidate_quality = assess_quality(
            matrices,
            selected_energies,
            selected_vectors,
            request.quality_policy,
        )
        quality = candidate_quality
        warnings: list[str] = []
        if linear_residual_warning:
            warnings.append("shifted linear residual exceeds diagnostic policy")
        if subspace_saturated:
            warnings.append(
                "CISS search subspace is saturated; the returned interval "
                "spectrum may be incomplete"
            )
        stages["solver_total"] = perf_counter() - total_started
        return IntervalSpectrum(
            energies_ev=selected_energies,
            quality=quality,
            backend=self.name,
            receipt=SolverReceipt(
                backend=self.name,
                device="cuda",
                scalar_dtype=complex_dtype.name,
                stage_seconds=stages,
                peak_gpu_memory_bytes=native_peak_gpu,
                counters={
                    "integration_points": self.settings.integration_points,
                    "contour_batch_size": batch_size,
                    "block_size": self.settings.block_size,
                    "moment_size": self.settings.moment_size,
                    "retained_subspace_rank": int(basis.shape[1]),
                    "selected_eigenpair_count": int(selected_energies.size),
                },
                details={
                    "native_stage_totals": native_stage_totals,
                    "maximum_relative_linear_residual": maximum_linear_residual,
                    "maximum_ghost_score": float(np.max(scores)),
                    "warnings": warnings,
                    "native_batches": native_batch_details,
                },
            ),
            complete=not subspace_saturated,
            eigenvectors=selected_vectors if request.retain_vectors else None,
            metadata={
                "algorithm": "ciss-contour-moments-rayleigh-ritz",
                "endpoint_guard_ev": self.settings.endpoint_guard_ev,
                "quality_status": quality.status,
                "warnings": warnings,
                "completeness_semantics": (
                    "completed contour solve; no formal eigenvalue-count proof"
                ),
            },
        )

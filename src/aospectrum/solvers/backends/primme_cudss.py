"""Indexed-state solver using cuDSS inertia and PRIMME interior solves."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from scipy import sparse

from aospectrum.assembly.pattern import SparseMatrixInput
from aospectrum.errors import ContractError, SolverError
from aospectrum.model.spectra import SolverReceipt
from aospectrum.solvers.indexed import (
    IndexedEigenpairs,
    IndexedSolveRequest,
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


class PrimmeTarget(str, Enum):
    """Direction relative to the current factor shift."""

    CLOSEST_LEQ = "closest_leq"
    CLOSEST_GEQ = "closest_geq"


@dataclass(frozen=True, slots=True)
class InertiaPoint:
    """Counts below, on and above one real shift."""

    energy_ev: float
    below: int
    zero: int
    above: int
    dimension: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_ev):
            raise ContractError("inertia energy must be finite")
        values = (self.below, self.zero, self.above, self.dimension)
        if any(isinstance(value, bool) or int(value) < 0 for value in values):
            raise ContractError("inertia counts must be nonnegative integers")
        if self.below + self.zero + self.above != self.dimension:
            raise ContractError("inertia counts must sum to matrix dimension")


@dataclass(frozen=True, slots=True)
class NumericUpdate:
    """One native factorization and inertia observation."""

    inertia: InertiaPoint
    numeric_epoch: int
    stage_seconds: Mapping[str, float]
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.inertia, InertiaPoint):
            raise ContractError("numeric update inertia is invalid")
        if isinstance(self.numeric_epoch, bool) or self.numeric_epoch <= 0:
            raise ContractError("numeric_epoch must be positive")
        stages = {str(key): float(value) for key, value in self.stage_seconds.items()}
        if not stages or any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in stages.items()
        ):
            raise ContractError("numeric update timings are invalid")
        object.__setattr__(self, "stage_seconds", MappingProxyType(stages))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True, slots=True, eq=False)
class PrimmeResult:
    """Native PRIMME eigenpairs relative to the current factor shift."""

    target: PrimmeTarget
    target_shift_ev: float
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    stage_seconds: Mapping[str, float]
    counters: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies_ev)
        vectors = np.asarray(self.eigenvectors)
        if energies.ndim != 1 or energies.dtype.kind != "f":
            raise ContractError("PRIMME energies must be one-dimensional real values")
        if energies.size == 0 or np.any(np.diff(energies) < 0.0):
            raise ContractError("PRIMME energies must be nonempty and sorted")
        if vectors.ndim != 2 or vectors.shape[1] != energies.size:
            raise ContractError("PRIMME eigenvector columns must match energies")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(vectors)):
            raise ContractError("PRIMME result must be finite")
        if not math.isfinite(self.target_shift_ev):
            raise ContractError("PRIMME target shift must be finite")
        if energies.flags.writeable:
            energies = np.array(energies, copy=True)
        if vectors.flags.writeable:
            vectors = np.array(vectors, copy=True, order="F")
        energies.setflags(write=False)
        vectors.setflags(write=False)
        stages = {str(key): float(value) for key, value in self.stage_seconds.items()}
        counters = {str(key): int(value) for key, value in self.counters.items()}
        object.__setattr__(self, "energies_ev", energies)
        object.__setattr__(self, "eigenvectors", vectors)
        object.__setattr__(self, "stage_seconds", MappingProxyType(stages))
        object.__setattr__(self, "counters", MappingProxyType(counters))
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@runtime_checkable
class PrimmeCudssSession(Protocol):
    """Native resources bound to one stable CSR pattern."""

    n_orbitals: int
    scalar_dtype: str

    def update_numeric(
        self,
        matrices: SparseMatrixInput,
        *,
        shift_ev: float,
    ) -> NumericUpdate: ...

    def solve(
        self,
        *,
        target: PrimmeTarget,
        count: int,
        tolerance: float,
        settings: "PrimmeSettings",
    ) -> PrimmeResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PrimmeSettings:
    """Versioned internal PRIMME and ordinal-location policy."""

    max_matvecs: int = 20_000
    max_basis_size: int = 256
    min_restart_size: int = 96
    max_block_size: int = 8
    initial_locator_step_ev: float = 0.25
    maximum_locator_expansions: int = 24
    maximum_locator_bisections: int = 32
    boundary_guard_ev: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in (
            "max_matvecs",
            "max_basis_size",
            "min_restart_size",
            "max_block_size",
            "maximum_locator_expansions",
            "maximum_locator_bisections",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) <= 0:
                raise ContractError(f"{name} must be a positive integer")
        if self.min_restart_size >= self.max_basis_size:
            raise ContractError("min_restart_size must be smaller than max_basis_size")
        for name in ("initial_locator_step_ev", "boundary_guard_ev"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ContractError(f"{name} must be positive and finite")

    def solver_tolerance(self, precision: str) -> float:
        return 1.0e-5 if precision == "float32" else 1.0e-10


def _scalar_dtype(precision: str) -> np.dtype:
    if precision == "float32":
        return np.dtype(np.float32)
    if precision == "float64":
        return np.dtype(np.float64)
    raise ContractError("precision must be 'float32' or 'float64'")


def _require_gamma_shared_pattern(
    matrices: SparseMatrixInput,
    precision: str,
) -> None:
    dtype = _scalar_dtype(precision)
    if matrices.hamiltonian.dtype != dtype or matrices.overlap.dtype != dtype:
        raise ContractError(
            f"primme-cudss requires Gamma-point {dtype.name} H/S"
        )
    if np.iscomplexobj(matrices.hamiltonian.data) or np.iscomplexobj(
        matrices.overlap.data
    ):
        raise ContractError("v1 Orbital solver accepts Gamma-point real H/S only")
    if not sparse.isspmatrix_csr(matrices.hamiltonian) or not sparse.isspmatrix_csr(
        matrices.overlap
    ):
        raise ContractError("primme-cudss requires CSR H/S")
    if not np.array_equal(
        matrices.hamiltonian.indptr,
        matrices.overlap.indptr,
    ) or not np.array_equal(
        matrices.hamiltonian.indices,
        matrices.overlap.indices,
    ):
        raise ContractError("primme-cudss requires one shared H/S CSR pattern")


def _pattern_token(matrices: SparseMatrixInput) -> tuple[Any, ...]:
    return (
        matrices.hamiltonian.shape,
        int(matrices.hamiltonian.indptr.__array_interface__["data"][0]),
        int(matrices.hamiltonian.indices.__array_interface__["data"][0]),
        matrices.hamiltonian.indptr.size,
        matrices.hamiltonian.indices.size,
    )


class NativePrimmeCudssSession:
    """Adapter around the optional unified CUDA extension."""

    def __init__(self, matrices: SparseMatrixInput, precision: str) -> None:
        if precision != "float32":
            raise SolverError(
                "primme-cudss v1 supports Gamma-point float32 only"
            )
        try:
            from aospectrum import _aospectrum_cuda
        except (ImportError, OSError) as exc:
            raise SolverError(
                "primme-cudss requires the optional AOSpectrum CUDA extension"
            ) from exc
        dtype = _scalar_dtype(precision)
        self.scalar_dtype = dtype.name
        self.n_orbitals = matrices.n_orbitals
        try:
            self._session = _aospectrum_cuda.PrimmeStateSession(
                matrices.hamiltonian.indptr,
                matrices.hamiltonian.indices,
                matrices.n_orbitals,
                self.scalar_dtype,
            )
        except RuntimeError as exc:
            raise SolverError(f"cannot create PRIMME/cuDSS session: {exc}") from exc

    def update_numeric(
        self,
        matrices: SparseMatrixInput,
        *,
        shift_ev: float,
    ) -> NumericUpdate:
        try:
            native = self._session.update_numeric(
                matrices.hamiltonian.data,
                matrices.overlap.data,
                shift_ev,
            )
        except RuntimeError as exc:
            raise SolverError(f"PRIMME/cuDSS numeric update failed: {exc}") from exc
        return NumericUpdate(
            inertia=InertiaPoint(
                energy_ev=shift_ev,
                below=int(native.below),
                zero=int(native.zero),
                above=int(native.above),
                dimension=self.n_orbitals,
            ),
            numeric_epoch=int(native.numeric_epoch),
            stage_seconds=dict(native.stage_seconds),
            details=dict(getattr(native, "details", {})),
        )

    def solve(
        self,
        *,
        target: PrimmeTarget,
        count: int,
        tolerance: float,
        settings: PrimmeSettings,
    ) -> PrimmeResult:
        try:
            native = self._session.solve(
                target.value,
                count,
                tolerance,
                settings.max_matvecs,
                settings.max_basis_size,
                settings.min_restart_size,
                min(settings.max_block_size, count),
            )
        except RuntimeError as exc:
            raise SolverError(f"PRIMME directed solve failed: {exc}") from exc
        return PrimmeResult(
            target=target,
            target_shift_ev=float(native.target_shift_ev),
            energies_ev=np.asarray(native.energies_ev),
            eigenvectors=np.asarray(native.eigenvectors),
            stage_seconds=dict(native.stage_seconds),
            counters=dict(getattr(native, "counters", {})),
            details=dict(getattr(native, "details", {})),
        )

    def close(self) -> None:
        try:
            self._session.close()
        except RuntimeError as exc:
            raise SolverError(f"cannot close PRIMME/cuDSS session: {exc}") from exc


@dataclass(frozen=True, slots=True)
class LocatorResult:
    anchor: NumericUpdate
    history: tuple[NumericUpdate, ...]


class OrdinalLocator:
    """Place the factor shift close enough to a requested ordinal window."""

    def __init__(self, settings: PrimmeSettings) -> None:
        self.settings = settings

    @staticmethod
    def _acceptable(update: NumericUpdate, first: int, last: int) -> bool:
        return update.inertia.zero == 0 and first - 1 <= update.inertia.below <= last

    def locate(
        self,
        session: PrimmeCudssSession,
        matrices: SparseMatrixInput,
        *,
        first: int,
        last: int,
        energy_hint_ev: float,
    ) -> LocatorResult:
        history: list[NumericUpdate] = []

        def observe(energy: float) -> NumericUpdate:
            update = session.update_numeric(matrices, shift_ev=energy)
            history.append(update)
            return update

        current = observe(energy_hint_ev)
        if self._acceptable(current, first, last):
            return LocatorResult(current, tuple(history))

        target = (first + last - 1) // 2
        lower: NumericUpdate | None = None
        upper: NumericUpdate | None = None
        if current.inertia.below < target:
            lower = current
            direction = 1.0
        else:
            upper = current
            direction = -1.0
        step = self.settings.initial_locator_step_ev
        for _ in range(self.settings.maximum_locator_expansions):
            current = observe(energy_hint_ev + direction * step)
            if self._acceptable(current, first, last):
                return LocatorResult(current, tuple(history))
            if current.inertia.below < target:
                lower = current
            else:
                upper = current
            if lower is not None and upper is not None:
                break
            step *= 2.0
        if lower is None or upper is None:
            raise SolverError("ordinal locator could not bracket the requested states")

        for _ in range(self.settings.maximum_locator_bisections):
            midpoint = 0.5 * (
                lower.inertia.energy_ev + upper.inertia.energy_ev
            )
            current = observe(midpoint)
            if self._acceptable(current, first, last):
                return LocatorResult(current, tuple(history))
            if current.inertia.below < target:
                lower = current
            else:
                upper = current
            if (
                upper.inertia.energy_ev - lower.inertia.energy_ev
                <= self.settings.boundary_guard_ev
            ):
                break
        candidate = min(
            (lower, upper),
            key=lambda item: abs(item.inertia.below - target),
        )
        if candidate.inertia.zero or not (
            first - 1 <= candidate.inertia.below <= last
        ):
            raise SolverError(
                "ordinal locator cannot place a nonsingular shift at the "
                "requested state boundary"
            )
        if history[-1] is not candidate:
            candidate = observe(candidate.inertia.energy_ev)
        return LocatorResult(candidate, tuple(history))


@dataclass(slots=True)
class PrimmeCudssSolver:
    """Directed partial-spectrum orchestration for Orbital states.

    The default session is PRIMME/cuDSS.  A compatible session may reuse the
    same ordinal-location and boundary-certification logic.
    """

    precision: str = "float32"
    settings: PrimmeSettings = field(default_factory=PrimmeSettings)
    native_session: PrimmeCudssSession | None = None
    name: str = "primme-cudss"
    receipt_device: str = "cuda"
    solve_stage_prefix: str = "primme"
    solve_counter_name: str = "primme_solves"
    solver_label: str = "PRIMME"
    factor_reuse_scope: str = "final-locator-shift"
    _pattern_token: tuple[Any, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_epoch: int = field(init=False, default=0, repr=False)

    def _session(self, matrices: SparseMatrixInput) -> PrimmeCudssSession:
        _require_gamma_shared_pattern(matrices, self.precision)
        token = _pattern_token(matrices)
        if self.native_session is None:
            self.native_session = NativePrimmeCudssSession(
                matrices,
                self.precision,
            )
            self._pattern_token = token
        elif self._pattern_token is None:
            self._pattern_token = token
        elif token != self._pattern_token:
            raise SolverError("PRIMME/cuDSS session cannot change its CSR pattern")
        expected_dtype = _scalar_dtype(self.precision).name
        if self.native_session.scalar_dtype != expected_dtype:
            raise SolverError("native PRIMME session scalar dtype differs")
        if self.native_session.n_orbitals != matrices.n_orbitals:
            raise SolverError("native PRIMME session dimension differs")
        return self.native_session

    def close(self) -> None:
        if self.native_session is not None:
            self.native_session.close()
            self.native_session = None
            self._pattern_token = None
            self._last_epoch = 0

    def _native_solve(
        self,
        session: PrimmeCudssSession,
        *,
        target: PrimmeTarget,
        count: int,
        request: IndexedSolveRequest,
        shift_ev: float,
    ) -> PrimmeResult:
        if count <= 0 or count >= session.n_orbitals:
            raise SolverError("PRIMME partial solve count is invalid")
        if count >= self.settings.max_basis_size:
            raise SolverError("requested state evidence exceeds PRIMME basis policy")
        result = session.solve(
            target=target,
            count=count,
            tolerance=self.settings.solver_tolerance(request.precision),
            settings=self.settings,
        )
        if not math.isclose(
            result.target_shift_ev,
            shift_ev,
            rel_tol=0.0,
            abs_tol=self.settings.boundary_guard_ev,
        ):
            raise SolverError("PRIMME result is not bound to the locator shift")
        if target is PrimmeTarget.CLOSEST_LEQ and np.any(
            result.energies_ev > shift_ev + self.settings.boundary_guard_ev
        ):
            raise SolverError(
                f"{self.solver_label} closest_leq returned an energy "
                "above its shift"
            )
        if target is PrimmeTarget.CLOSEST_GEQ and np.any(
            result.energies_ev < shift_ev - self.settings.boundary_guard_ev
        ):
            raise SolverError(
                f"{self.solver_label} closest_geq returned an energy "
                "below its shift"
            )
        return result

    def solve_states(
        self,
        matrices: SparseMatrixInput,
        request: IndexedSolveRequest,
    ) -> IndexedEigenpairs:
        if request.precision != self.precision:
            raise SolverError("indexed request precision differs from backend")
        if request.selection.last > matrices.n_orbitals:
            raise SolverError("requested state range exceeds matrix dimension")
        session = self._session(matrices)
        started = perf_counter()
        locator = OrdinalLocator(self.settings).locate(
            session,
            matrices,
            first=request.selection.first,
            last=request.selection.last,
            energy_hint_ev=(
                0.0 if request.energy_hint_ev is None else request.energy_hint_ev
            ),
        )
        for update in locator.history:
            if update.numeric_epoch <= self._last_epoch:
                raise SolverError("native numeric epoch did not increase")
            self._last_epoch = update.numeric_epoch
        anchor = locator.anchor.inertia
        evidence_first = max(1, request.selection.first - 1)
        evidence_last = min(matrices.n_orbitals, request.selection.last + 1)
        lower_count = (
            anchor.below - evidence_first + 1
            if evidence_first <= anchor.below
            else 0
        )
        upper_count = (
            evidence_last - anchor.below
            if evidence_last > anchor.below
            else 0
        )
        raw_results: list[PrimmeResult] = []
        energies: list[np.ndarray] = []
        vectors: list[np.ndarray] = []
        states: list[np.ndarray] = []
        if lower_count:
            result = self._native_solve(
                session,
                target=PrimmeTarget.CLOSEST_LEQ,
                count=lower_count,
                request=request,
                shift_ev=anchor.energy_ev,
            )
            raw_results.append(result)
            energies.append(result.energies_ev)
            vectors.append(result.eigenvectors)
            states.append(
                np.arange(
                    anchor.below - lower_count + 1,
                    anchor.below + 1,
                    dtype=np.int64,
                )
            )
        if upper_count:
            result = self._native_solve(
                session,
                target=PrimmeTarget.CLOSEST_GEQ,
                count=upper_count,
                request=request,
                shift_ev=anchor.energy_ev,
            )
            raw_results.append(result)
            energies.append(result.energies_ev)
            vectors.append(result.eigenvectors)
            states.append(
                np.arange(
                    anchor.below + 1,
                    anchor.below + upper_count + 1,
                    dtype=np.int64,
                )
            )
        returned_states = np.concatenate(states)
        returned_energies = np.concatenate(energies)
        returned_vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(returned_states, kind="stable")
        returned_states = returned_states[order]
        returned_energies = returned_energies[order]
        returned_vectors = returned_vectors[:, order]
        mask = np.logical_and(
            returned_states >= request.selection.first,
            returned_states <= request.selection.last,
        )
        selected_states = returned_states[mask]
        expected = np.arange(
            request.selection.first,
            request.selection.last + 1,
            dtype=np.int64,
        )
        if not np.array_equal(selected_states, expected):
            raise SolverError(
                f"directed {self.solver_label} solves missed requested states"
            )
        state_position = {
            int(state): index for index, state in enumerate(returned_states)
        }
        warnings: list[str] = []
        for boundary, neighbor in (
            (request.selection.first, request.selection.first - 1),
            (request.selection.last, request.selection.last + 1),
        ):
            if neighbor < 1 or neighbor > matrices.n_orbitals:
                continue
            gap = abs(
                returned_energies[state_position[boundary]]
                - returned_energies[state_position[neighbor]]
            )
            if gap <= self.settings.boundary_guard_ev:
                warnings.append(
                    "requested state boundary splits a near-degenerate "
                    f"subspace at states {min(boundary, neighbor)} and "
                    f"{max(boundary, neighbor)} (gap={gap:.6e} eV); "
                    "the individual orbital may depend on the solver basis"
                )
        selected_energies = returned_energies[mask]
        selected_vectors = np.asfortranarray(returned_vectors[:, mask])
        quality = assess_quality(
            matrices,
            selected_energies,
            selected_vectors,
            request.quality_policy,
        )
        stage_seconds: dict[str, float] = {
            "indexed_backend_total": perf_counter() - started
        }
        for update in locator.history:
            for name, seconds in update.stage_seconds.items():
                stage_seconds[f"inertia_{name}"] = (
                    stage_seconds.get(f"inertia_{name}", 0.0) + seconds
                )
        for result in raw_results:
            for name, seconds in result.stage_seconds.items():
                key = f"{self.solve_stage_prefix}_{name}"
                stage_seconds[key] = (
                    stage_seconds.get(key, 0.0) + seconds
                )
        receipt_counters = {
            "inertia_factorizations": len(locator.history),
            self.solve_counter_name: len(raw_results),
            "selected_eigenpair_count": int(selected_states.size),
        }
        for counter_name in ("matvecs", "iterations"):
            if any(
                counter_name in result.counters
                for result in raw_results
            ):
                receipt_counters[counter_name] = sum(
                    result.counters.get(counter_name, 0)
                    for result in raw_results
                )
        update_details = [
            _receipt_value(dict(update.details))
            for update in locator.history
        ]
        solve_details = [
            _receipt_value(dict(result.details))
            for result in raw_results
        ]
        peak_device_values = [
            int(details["peak_device_bytes"])
            for details in (*update_details, *solve_details)
            if isinstance(details, Mapping)
            and details.get("peak_device_bytes") is not None
        ]
        return IndexedEigenpairs(
            absolute_state_numbers=selected_states,
            energies_ev=selected_energies,
            eigenvectors=selected_vectors,
            quality=quality,
            backend=self.name,
            receipt=SolverReceipt(
                backend=self.name,
                device=self.receipt_device,
                scalar_dtype=_scalar_dtype(self.precision).name,
                stage_seconds=stage_seconds,
                peak_gpu_memory_bytes=(
                    max(peak_device_values)
                    if peak_device_values
                    else None
                ),
                counters=receipt_counters,
                details={
                    "factor_reuse_scope": self.factor_reuse_scope,
                    "warnings": warnings,
                    "numeric_updates": update_details,
                    "directed_solves": solve_details,
                },
            ),
            locator_evidence={
                "anchor_energy_ev": anchor.energy_ev,
                "anchor_below": anchor.below,
                "anchor_zero": anchor.zero,
                "history": [
                    {
                        "energy_ev": update.inertia.energy_ev,
                        "below": update.inertia.below,
                        "zero": update.inertia.zero,
                    }
                    for update in locator.history
                ],
            },
            metadata={
                "quality_status": quality.status,
                "warnings": warnings,
            },
        )

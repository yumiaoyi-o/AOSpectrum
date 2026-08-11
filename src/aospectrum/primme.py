"""Float32 Gamma indexed-state solver using cuDSS and PRIMME."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from .data import StateRange
from .errors import InputError, SolverError
from .sparse import SparsePair


HARTREE_TO_EV = 27.211386245988


class PrimmeTarget(str, Enum):
    """Direction relative to the current factor shift."""

    CLOSEST_LEQ = "closest_leq"
    CLOSEST_GEQ = "closest_geq"


@dataclass(frozen=True, slots=True)
class InertiaPoint:
    energy_ev: float
    below: int
    zero: int
    above: int
    dimension: int

@dataclass(slots=True)
class NumericUpdate:
    inertia: InertiaPoint
    numeric_epoch: int
    stage_seconds: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrimmeResult:
    target: PrimmeTarget
    target_shift_ev: float
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    stage_seconds: dict[str, float]
    counters: dict[str, int] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexedResult:
    state_numbers: np.ndarray
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    maximum_residual_ev: float
    stage_seconds: dict[str, float]
    peak_gpu_memory_bytes: int | None
    warnings: tuple[str, ...]
    locator_history: tuple[InertiaPoint, ...]


@dataclass(frozen=True, slots=True)
class CudaSparsePair:
    """Borrowed device CSR values with an immutable host topology.

    The value pointers remain owned by the caller until synchronous device
    admission returns. AOSpectrum then operates on its own resident buffers.
    """

    indptr: np.ndarray
    indices: np.ndarray
    n_orbitals: int
    hamiltonian_values_pointer: int
    overlap_values_pointer: int
    nnz: int
    producer_stream_pointer: int
    hamiltonian_scale_to_ev: float = 1.0
    hamiltonian_values_owner: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    overlap_values_owner: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.n_orbitals <= 0 or self.nnz <= 0:
            raise InputError("CUDA CSR dimensions must be positive")
        if self.indptr.dtype != np.int32 or self.indices.dtype != np.int32:
            raise InputError("CUDA CSR topology must use host int32 arrays")
        if self.indptr.ndim != 1 or self.indices.ndim != 1:
            raise InputError("CUDA CSR topology must be one-dimensional")
        if self.indptr.size != self.n_orbitals + 1:
            raise InputError("CUDA CSR row offsets differ from matrix size")
        if self.indices.size != self.nnz:
            raise InputError("CUDA CSR columns differ from declared nnz")
        if int(self.indptr[0]) != 0 or int(self.indptr[-1]) != self.nnz:
            raise InputError("CUDA CSR row offsets do not span nnz")
        if self.hamiltonian_values_pointer <= 0 or self.overlap_values_pointer <= 0:
            raise InputError("CUDA H/S value pointers must be nonzero")
        if not math.isfinite(self.hamiltonian_scale_to_ev) or (
            self.hamiltonian_scale_to_ev <= 0.0
        ):
            raise InputError("CUDA Hamiltonian energy scale must be positive")


@dataclass(frozen=True, slots=True)
class PrimmeSettings:
    max_matvecs: int = 20_000
    max_basis_size: int = 256
    min_restart_size: int = 96
    max_block_size: int = 8
    initial_locator_step_ev: float = 0.25
    maximum_locator_expansions: int = 24
    maximum_locator_bisections: int = 32
    boundary_guard_ev: float = 1.0e-5

    solver_tolerance: float = 1.0e-5


def _require_gamma_shared_pattern(matrices: SparsePair) -> None:
    if (
        matrices.hamiltonian.dtype != np.float32
        or matrices.overlap.dtype != np.float32
    ):
        raise InputError("Orbital requires Gamma-point float32 H/S")
    if not sparse.isspmatrix_csr(matrices.hamiltonian) or not sparse.isspmatrix_csr(
        matrices.overlap
    ):
        raise InputError("Orbital requires CSR H/S")
    if not np.array_equal(
        matrices.hamiltonian.indptr,
        matrices.overlap.indptr,
    ) or not np.array_equal(
        matrices.hamiltonian.indices,
        matrices.overlap.indices,
    ):
        raise InputError("Orbital requires one shared H/S CSR pattern")


class NativePrimmeCudssSession:
    def __init__(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        n_orbitals: int,
    ) -> None:
        try:
            from aospectrum import _aospectrum_cuda
        except (ImportError, OSError) as exc:
            raise SolverError(
                "primme-cudss requires the optional AOSpectrum CUDA extension"
            ) from exc
        self.n_orbitals = int(n_orbitals)
        try:
            self._session = _aospectrum_cuda.PrimmeStateSession(
                indptr,
                indices,
                self.n_orbitals,
                "float32",
            )
        except RuntimeError as exc:
            raise SolverError(f"cannot create PRIMME/cuDSS session: {exc}") from exc

    def update_numeric(
        self,
        matrices: SparsePair,
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

    def load_device_values(self, matrices: CudaSparsePair) -> dict[str, float]:
        try:
            native = self._session.load_device_values(
                matrices.hamiltonian_values_pointer,
                matrices.overlap_values_pointer,
                matrices.nnz,
                matrices.producer_stream_pointer,
                matrices.hamiltonian_scale_to_ev,
            )
        except RuntimeError as exc:
            raise SolverError(
                f"PRIMME/cuDSS device admission failed: {exc}"
            ) from exc
        return dict(native.stage_seconds)

    def factor_resident(self, *, shift_ev: float) -> NumericUpdate:
        try:
            native = self._session.factor_resident(shift_ev)
        except RuntimeError as exc:
            raise SolverError(
                f"PRIMME/cuDSS resident factorization failed: {exc}"
            ) from exc
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
        initial_vectors: np.ndarray | None = None,
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
                initial_vectors,
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
        session: Any,
        matrices: SparsePair,
        *,
        first: int,
        last: int,
        energy_hint_ev: float,
    ) -> LocatorResult:
        return self.locate_observing(
            lambda energy: session.update_numeric(matrices, shift_ev=energy),
            first=first,
            last=last,
            energy_hint_ev=energy_hint_ev,
        )

    def locate_resident(
        self,
        session: Any,
        *,
        first: int,
        last: int,
        energy_hint_ev: float,
    ) -> LocatorResult:
        return self.locate_observing(
            lambda energy: session.factor_resident(shift_ev=energy),
            first=first,
            last=last,
            energy_hint_ev=energy_hint_ev,
        )

    def locate_observing(
        self,
        observer: Any,
        *,
        first: int,
        last: int,
        energy_hint_ev: float,
    ) -> LocatorResult:
        history: list[NumericUpdate] = []

        def observe(energy: float) -> NumericUpdate:
            update = observer(energy)
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
    """Locate and solve an absolute Gamma-point state range."""

    settings: PrimmeSettings = field(default_factory=PrimmeSettings)
    native_session: Any | None = None
    _pattern_token: tuple[Any, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pattern_indptr: np.ndarray | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pattern_indices: np.ndarray | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pattern_source_indptr: np.ndarray | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _pattern_source_indices: np.ndarray | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _last_epoch: int = field(init=False, default=0, repr=False)

    def _session_for_topology(
        self,
        indptr: np.ndarray,
        indices: np.ndarray,
        n_orbitals: int,
    ) -> Any:
        token = (
            int(n_orbitals),
            int(indptr.__array_interface__["data"][0]),
            int(indices.__array_interface__["data"][0]),
            indptr.size,
            indices.size,
        )
        if self.native_session is None:
            self.native_session = NativePrimmeCudssSession(
                indptr,
                indices,
                n_orbitals,
            )
            self._pattern_token = token
            self._pattern_indptr = indptr.copy()
            self._pattern_indices = indices.copy()
            self._pattern_source_indptr = indptr
            self._pattern_source_indices = indices
        elif self._pattern_token is None:
            self._pattern_token = token
            self._pattern_indptr = indptr.copy()
            self._pattern_indices = indices.copy()
            self._pattern_source_indptr = indptr
            self._pattern_source_indices = indices
        elif token != self._pattern_token:
            same_pattern = (
                self._pattern_source_indptr is indptr
                and self._pattern_source_indices is indices
            ) or (
                self._pattern_indptr is not None
                and self._pattern_indices is not None
                and np.array_equal(
                    self._pattern_indptr,
                    indptr,
                )
                and np.array_equal(
                    self._pattern_indices,
                    indices,
                )
            )
            if same_pattern:
                self._pattern_token = token
                self._pattern_source_indptr = indptr
                self._pattern_source_indices = indices
            else:
                self.close()
                self.native_session = NativePrimmeCudssSession(
                    indptr,
                    indices,
                    n_orbitals,
                )
                self._pattern_token = token
                self._pattern_indptr = indptr.copy()
                self._pattern_indices = indices.copy()
                self._pattern_source_indptr = indptr
                self._pattern_source_indices = indices
        if self.native_session.n_orbitals != n_orbitals:
            raise SolverError("native PRIMME session dimension differs")
        return self.native_session

    def _session(self, matrices: SparsePair) -> Any:
        _require_gamma_shared_pattern(matrices)
        return self._session_for_topology(
            matrices.hamiltonian.indptr,
            matrices.hamiltonian.indices,
            matrices.n_orbitals,
        )

    def close(self) -> None:
        if self.native_session is not None:
            self.native_session.close()
            self.native_session = None
            self._pattern_token = None
            self._pattern_indptr = None
            self._pattern_indices = None
            self._pattern_source_indptr = None
            self._pattern_source_indices = None
            self._last_epoch = 0

    def _native_solve(
        self,
        session: Any,
        *,
        target: PrimmeTarget,
        count: int,
        shift_ev: float,
        initial_vectors: np.ndarray | None = None,
    ) -> PrimmeResult:
        if count <= 0 or count >= session.n_orbitals:
            raise SolverError("PRIMME partial solve count is invalid")
        if count >= self.settings.max_basis_size:
            raise SolverError("requested state range exceeds the PRIMME basis size")
        result = session.solve(
            target=target,
            count=count,
            tolerance=self.settings.solver_tolerance,
            settings=self.settings,
            initial_vectors=initial_vectors,
        )
        order = np.argsort(result.energies_ev, kind="stable")
        result.energies_ev = np.asarray(result.energies_ev)[order]
        result.eigenvectors = np.asfortranarray(
            np.asarray(result.eigenvectors)[:, order],
            dtype=np.float32,
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
            raise SolverError("PRIMME closest_leq returned an energy above its shift")
        if target is PrimmeTarget.CLOSEST_GEQ and np.any(
            result.energies_ev < shift_ev - self.settings.boundary_guard_ev
        ):
            raise SolverError("PRIMME closest_geq returned an energy below its shift")
        return result

    @staticmethod
    def _initial_vectors(
        initial: IndexedResult | None,
        target_states: np.ndarray,
        n_orbitals: int,
    ) -> np.ndarray | None:
        if initial is None or initial.eigenvectors.shape[0] != n_orbitals:
            return None
        positions = {
            int(state): index
            for index, state in enumerate(initial.state_numbers)
        }
        columns = [
            positions[int(state)]
            for state in target_states
            if int(state) in positions
        ]
        if not columns:
            return None
        return np.asfortranarray(
            initial.eigenvectors[:, columns],
            dtype=np.float32,
        )

    def solve_states(
        self,
        matrices: SparsePair,
        selection: StateRange,
        *,
        energy_hint_ev: float = 0.0,
        initial: IndexedResult | None = None,
    ) -> IndexedResult:
        if selection.last > matrices.n_orbitals:
            raise SolverError("requested state range exceeds matrix dimension")
        session = self._session(matrices)
        started = perf_counter()
        locator = OrdinalLocator(self.settings).locate(
            session,
            matrices,
            first=selection.first,
            last=selection.last,
            energy_hint_ev=energy_hint_ev,
        )
        return self._finish_state_solve(
            session=session,
            n_orbitals=matrices.n_orbitals,
            selection=selection,
            locator=locator,
            started=started,
            initial=initial,
            admission_stage_seconds=None,
            residual_evaluator=lambda energies, vectors, _results: float(
                np.max(
                    np.linalg.norm(
                        matrices.hamiltonian @ vectors
                        - (matrices.overlap @ vectors) * energies[None, :],
                        axis=0,
                    ),
                    initial=0.0,
                )
            ),
        )

    def solve_states_device(
        self,
        matrices: CudaSparsePair,
        selection: StateRange,
        *,
        energy_hint_ev: float = 0.0,
        initial: IndexedResult | None = None,
    ) -> IndexedResult:
        admission_stage_seconds = self.admit_device_values(matrices)
        return self.solve_admitted_states(
            StateRange(selection.first, selection.last),
            n_orbitals=matrices.n_orbitals,
            energy_hint_ev=energy_hint_ev,
            initial=initial,
            admission_stage_seconds=admission_stage_seconds,
        )

    def admit_device_values(
        self,
        matrices: CudaSparsePair,
    ) -> dict[str, float]:
        session = self._session_for_topology(
            matrices.indptr,
            matrices.indices,
            matrices.n_orbitals,
        )
        return session.load_device_values(matrices)

    def solve_admitted_states(
        self,
        selection: StateRange,
        *,
        n_orbitals: int,
        energy_hint_ev: float = 0.0,
        initial: IndexedResult | None = None,
        admission_stage_seconds: dict[str, float] | None = None,
    ) -> IndexedResult:
        if self.native_session is None:
            raise SolverError("no device-resident H/S values were admitted")
        if selection.last > n_orbitals:
            raise SolverError("requested state range exceeds matrix dimension")
        session = self.native_session
        if session.n_orbitals != n_orbitals:
            raise SolverError("admitted matrix dimension differs")
        started = perf_counter()
        locator = OrdinalLocator(self.settings).locate_resident(
            session,
            first=selection.first,
            last=selection.last,
            energy_hint_ev=energy_hint_ev,
        )

        def native_residual_ev(
            _energies: np.ndarray,
            _vectors: np.ndarray,
            results: list[PrimmeResult],
        ) -> float:
            residuals = [
                np.asarray(result.details["residual_norms_hartree"])
                for result in results
                if "residual_norms_hartree" in result.details
            ]
            if len(residuals) != len(results):
                raise SolverError("native CUDA solve omitted eigenpair residuals")
            return float(
                np.max(
                    np.concatenate(residuals),
                    initial=0.0,
                )
                * HARTREE_TO_EV
            )

        return self._finish_state_solve(
            session=session,
            n_orbitals=n_orbitals,
            selection=selection,
            locator=locator,
            started=started,
            initial=initial,
            admission_stage_seconds=admission_stage_seconds,
            residual_evaluator=native_residual_ev,
        )

    def _finish_state_solve(
        self,
        *,
        session: Any,
        n_orbitals: int,
        selection: StateRange,
        locator: LocatorResult,
        started: float,
        initial: IndexedResult | None,
        admission_stage_seconds: dict[str, float] | None,
        residual_evaluator: Any,
    ) -> IndexedResult:
        for update in locator.history:
            if update.numeric_epoch <= self._last_epoch:
                raise SolverError("native numeric epoch did not increase")
            self._last_epoch = update.numeric_epoch
        anchor = locator.anchor.inertia
        evidence_first = max(1, selection.first - 1)
        evidence_last = min(n_orbitals, selection.last + 1)
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
            lower_states = np.arange(
                anchor.below - lower_count + 1,
                anchor.below + 1,
                dtype=np.int64,
            )
            result = self._native_solve(
                session,
                target=PrimmeTarget.CLOSEST_LEQ,
                count=lower_count,
                shift_ev=anchor.energy_ev,
                initial_vectors=self._initial_vectors(
                    initial,
                    lower_states,
                    n_orbitals,
                ),
            )
            raw_results.append(result)
            energies.append(result.energies_ev)
            vectors.append(result.eigenvectors)
            states.append(lower_states)
        if upper_count:
            upper_states = np.arange(
                anchor.below + 1,
                anchor.below + upper_count + 1,
                dtype=np.int64,
            )
            result = self._native_solve(
                session,
                target=PrimmeTarget.CLOSEST_GEQ,
                count=upper_count,
                shift_ev=anchor.energy_ev,
                initial_vectors=self._initial_vectors(
                    initial,
                    upper_states,
                    n_orbitals,
                ),
            )
            raw_results.append(result)
            energies.append(result.energies_ev)
            vectors.append(result.eigenvectors)
            states.append(upper_states)
        returned_states = np.concatenate(states)
        returned_energies = np.concatenate(energies)
        returned_vectors = np.concatenate(vectors, axis=1)
        order = np.argsort(returned_states, kind="stable")
        returned_states = returned_states[order]
        returned_energies = returned_energies[order]
        returned_vectors = returned_vectors[:, order]
        mask = np.logical_and(
            returned_states >= selection.first,
            returned_states <= selection.last,
        )
        selected_states = returned_states[mask]
        expected = np.arange(
            selection.first,
            selection.last + 1,
            dtype=np.int64,
        )
        if not np.array_equal(selected_states, expected):
            raise SolverError("directed PRIMME solves missed requested states")
        state_position = {
            int(state): index for index, state in enumerate(returned_states)
        }
        warnings: list[str] = []
        for boundary, neighbor in (
            (selection.first, selection.first - 1),
            (selection.last, selection.last + 1),
        ):
            if neighbor < 1 or neighbor > n_orbitals:
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
        selected_vectors = np.asfortranarray(
            returned_vectors[:, mask],
            dtype=np.float32,
        )
        maximum_residual_ev = residual_evaluator(
            selected_energies,
            selected_vectors,
            raw_results,
        )
        if maximum_residual_ev > 1.0e-3:
            warnings.append(
                "maximum eigenpair residual is "
                f"{maximum_residual_ev:.3e} eV"
            )
        stage_seconds: dict[str, float] = {
            "indexed_backend_total": perf_counter() - started
        }
        if admission_stage_seconds is not None:
            stage_seconds.update(admission_stage_seconds)
        for update in locator.history:
            for name, seconds in update.stage_seconds.items():
                stage_seconds[f"inertia_{name}"] = (
                    stage_seconds.get(f"inertia_{name}", 0.0) + seconds
                )
        for result in raw_results:
            for name, seconds in result.stage_seconds.items():
                key = f"primme_{name}"
                stage_seconds[key] = (
                    stage_seconds.get(key, 0.0) + seconds
                )
        detail_sets = [
            item.details
            for item in (*locator.history, *raw_results)
        ]
        peak_device_values = [
            int(details["peak_device_bytes"])
            for details in detail_sets
            if details.get("peak_device_bytes") is not None
        ]
        return IndexedResult(
            state_numbers=selected_states,
            energies_ev=np.asarray(selected_energies, dtype=np.float32),
            eigenvectors=selected_vectors,
            maximum_residual_ev=maximum_residual_ev,
            stage_seconds=stage_seconds,
            peak_gpu_memory_bytes=(
                max(peak_device_values) if peak_device_values else None
            ),
            warnings=tuple(warnings),
            locator_history=tuple(update.inertia for update in locator.history),
        )

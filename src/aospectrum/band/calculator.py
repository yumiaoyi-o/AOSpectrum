"""Band application service built on the interval-solver contract."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from aospectrum.assembly.pattern import BlochPattern, BlochWorkspace
from aospectrum.band.model import (
    BAND_CONTINUITY_GUARD_EV_V1,
    BandPoint,
    BandRequest,
    BandResult,
)
from aospectrum.errors import SolverError
from aospectrum.model.operators import LocalizedOperatorBundle
from aospectrum.model.spectra import EnergyInterval
from aospectrum.solvers.interval import (
    EnergyIntervalSolver,
    IntervalSolveRequest,
)


@dataclass(slots=True)
class BandCalculator:
    """Own one reusable sparse workspace and one interval-solver session."""

    solver: EnergyIntervalSolver

    def calculate(
        self,
        bundle: LocalizedOperatorBundle,
        request: BandRequest,
    ) -> BandResult:
        prepared = self.prepare(bundle, request)
        points = tuple(
            prepared.solve_point(index)
            for index in range(request.kpath.n_points)
        )
        return prepared.build_result(points)

    def prepare(
        self,
        bundle: LocalizedOperatorBundle,
        request: BandRequest,
    ) -> "PreparedBandCalculation":
        reference_ev = request.energy_reference.resolve(bundle)
        solve_interval_ev = request.display_interval_ev.expanded(
            BAND_CONTINUITY_GUARD_EV_V1
        )
        absolute_interval = EnergyInterval(
            solve_interval_ev.lower_ev + reference_ev,
            solve_interval_ev.upper_ev + reference_ev,
        )
        pattern_started = perf_counter()
        pattern = BlochPattern.from_bundle(bundle)
        complex_dtype = np.dtype(
            np.complex64
            if request.precision == "float32"
            else np.complex128
        )
        workspace = pattern.create_workspace(complex_dtype)
        pattern_seconds = perf_counter() - pattern_started
        return PreparedBandCalculation(
            bundle=bundle,
            request=request,
            solver=self.solver,
            pattern=pattern,
            workspace=workspace,
            reference_energy_ev=reference_ev,
            solve_interval_ev=solve_interval_ev,
            absolute_solve_interval_ev=absolute_interval,
            pattern_seconds=pattern_seconds,
        )

    def close(self) -> None:
        close = getattr(self.solver, "close", None)
        if callable(close):
            close()


@dataclass(slots=True)
class PreparedBandCalculation:
    """Stable topology/workspace shared by serial and sharded executors."""

    bundle: LocalizedOperatorBundle
    request: BandRequest
    solver: EnergyIntervalSolver
    pattern: BlochPattern
    workspace: BlochWorkspace
    reference_energy_ev: float
    solve_interval_ev: EnergyInterval
    absolute_solve_interval_ev: EnergyInterval
    pattern_seconds: float

    def solve_point(self, index: int) -> BandPoint:
        if isinstance(index, bool) or not 0 <= int(index) < self.request.kpath.n_points:
            raise SolverError("Band point index is out of range")
        normalized = int(index)
        kpoint = self.request.kpath.fractional_points[normalized]
        assembly_started = perf_counter()
        phase = self.pattern.prepare_phase(
            self.bundle,
            kpoint,
            precision=self.request.precision,
            force_complex=True,
        )
        matrices = self.workspace.update(self.bundle, phase)
        assembly_seconds = perf_counter() - assembly_started
        spectrum = self.solver.solve_interval(
            matrices,
            IntervalSolveRequest(
                interval=self.absolute_solve_interval_ev,
                precision=self.request.precision,
                retain_vectors=False,
            ),
        )
        if not spectrum.complete:
            raise SolverError(
                f"interval backend returned incomplete point {normalized}"
            )
        return BandPoint(
            index=normalized,
            k_fractional=kpoint,
            absolute_energies_ev=spectrum.energies_ev,
            quality=spectrum.quality,
            receipt=spectrum.receipt,
            assembly_metadata={
                "assembly_seconds": assembly_seconds,
                "workspace_generation": matrices.generation,
                "h_error_before_projection": (
                    matrices.h_error_before_projection
                ),
                "s_error_before_projection": (
                    matrices.s_error_before_projection
                ),
                "h_projection_tolerance": matrices.h_projection_tolerance,
                "s_projection_tolerance": matrices.s_projection_tolerance,
            },
        )

    def build_result(self, points: Iterable[BandPoint]) -> BandResult:
        ordered = tuple(sorted(points, key=lambda point: point.index))
        scope_warnings: list[str] = []
        if self.bundle.basis.spinor_width != 1:
            scope_warnings.append(
                "spinor_width differs from the validated scalar-orbital "
                "scope; band interpretation may be inaccurate"
            )
        if not all(self.bundle.structure.periodic):
            scope_warnings.append(
                "non-periodic axes are outside the validated periodic "
                "Bloch convention; band interpretation may be inaccurate"
            )
        return BandResult(
            kpath=self.request.kpath,
            display_interval_ev=self.request.display_interval_ev,
            solve_interval_ev=self.solve_interval_ev,
            reference_energy_ev=self.reference_energy_ev,
            points=ordered,
            precision=self.request.precision,
            backend=self.solver.name,
            metadata={
                "pattern_seconds": self.pattern_seconds,
                "pattern_nnz": self.pattern.nnz,
                "pattern_contributions": self.pattern.contribution_count,
                "workspace_bytes": self.workspace.value_nbytes,
                "cell_angstrom": self.bundle.structure.cell_angstrom.tolist(),
                "warnings": scope_warnings,
                "quality_status": (
                    "quality_warning"
                    if any(point.quality.warnings for point in ordered)
                    else "passed"
                ),
            },
        )

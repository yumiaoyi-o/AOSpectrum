"""Selected-state Orbital application service."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aospectrum.assembly.pattern import BlochPattern
from aospectrum.errors import SolverError
from aospectrum.model.operators import LocalizedOperatorBundle
from aospectrum.orbital.model import (
    OrbitalEigensystem,
    OrbitalRequest,
    OrbitalStateField,
)
from aospectrum.orbital.providers.base import (
    BasisFunctionProvider,
    FieldResources,
)
from aospectrum.orbital.selection import resolve_state_selection
from aospectrum.solvers.indexed import IndexedSolveRequest, IndexedStateSolver


def _phase_fix(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(vectors)
    complex_dtype = (
        np.complex64
        if source.dtype in {np.dtype(np.float32), np.dtype(np.complex64)}
        else np.complex128
    )
    fixed = np.array(source, dtype=complex_dtype, copy=True, order="F")
    anchors = np.argmax(np.abs(fixed), axis=0).astype(np.int64)
    for column, anchor in enumerate(anchors):
        value = fixed[int(anchor), column]
        if value == 0.0:
            raise SolverError("cannot phase-fix a zero eigenvector")
        phase = np.exp(-1j * np.angle(value))
        fixed[:, column] *= phase
        if np.real(fixed[int(anchor), column]) < 0.0:
            fixed[:, column] *= -1.0
    return fixed, anchors


@dataclass(slots=True)
class OrbitalCalculator:
    """Own the indexed-state backend for one selected-state calculation."""

    solver: IndexedStateSolver

    def solve_eigensystem(
        self,
        bundle: LocalizedOperatorBundle,
        request: OrbitalRequest,
    ) -> OrbitalEigensystem:
        selection = resolve_state_selection(request.states, bundle.filling)
        pattern = BlochPattern.from_bundle(bundle)
        phase = pattern.prepare_phase(
            bundle,
            (0.0, 0.0, 0.0),
            precision=request.precision,
        )
        matrices = pattern.create_workspace(phase.dtype).update(bundle, phase)
        if np.iscomplexobj(matrices.hamiltonian.data):
            raise SolverError(
                "v1 Orbital accepts only a real Gamma-point generalized problem"
            )
        hint = request.energy_hint_ev
        if (
            hint is None
            and bundle.filling is not None
            and bundle.filling.fermi_energy_ev is not None
        ):
            hint = bundle.filling.fermi_energy_ev
        solved = self.solver.solve_states(
            matrices,
            IndexedSolveRequest(
                selection=selection.absolute,
                precision=request.precision,
                energy_hint_ev=hint,
            ),
        )
        fixed, anchors = _phase_fix(solved.eigenvectors)
        notices = [
            str(value)
            for value in solved.metadata.get("warnings", ())
        ]
        if bundle.basis.spinor_width != 1:
            notices.append(
                "spinor_width differs from the validated scalar-orbital "
                "scope; rendered orbitals may be inaccurate"
            )
        for index, gap in enumerate(np.diff(solved.energies_ev)):
            if abs(float(gap)) <= request.degeneracy_notice_ev:
                left = int(solved.absolute_state_numbers[index])
                right = int(solved.absolute_state_numbers[index + 1])
                notices.append(
                    f"states {left} and {right} are near-degenerate "
                    f"(gap={abs(float(gap)):.6e} eV)"
                )
        return OrbitalEigensystem(
            selection=selection,
            absolute_state_numbers=solved.absolute_state_numbers,
            energies_ev=solved.energies_ev,
            eigenvectors=fixed,
            phase_anchor_indices=anchors,
            quality=solved.quality,
            receipt=solved.receipt,
            backend=solved.backend,
            degeneracy_notices=tuple(notices),
            locator_evidence=solved.locator_evidence,
            metadata={
                "k_fractional": [0.0, 0.0, 0.0],
                "phase_policy": "maximum-coefficient-positive-real/v1",
                "quality_status": solved.quality.status,
            },
        )

    def iter_fields(
        self,
        bundle: LocalizedOperatorBundle,
        request: OrbitalRequest,
        eigensystem: OrbitalEigensystem,
        provider: BasisFunctionProvider,
        resources: FieldResources,
    ):
        """Yield one field at a time so large volumes are never accumulated."""

        if provider.basis_identity != bundle.basis.basis_identity:
            raise SolverError("basis provider differs from solved AO layout")
        for column, state_number in enumerate(
            eigensystem.absolute_state_numbers
        ):
            yield self.evaluate_state(
                bundle,
                request,
                eigensystem,
                provider,
                resources,
                column=column,
            )

    def evaluate_state(
        self,
        bundle: LocalizedOperatorBundle,
        request: OrbitalRequest,
        eigensystem: OrbitalEigensystem,
        provider: BasisFunctionProvider,
        resources: FieldResources,
        *,
        column: int,
    ) -> OrbitalStateField:
        """Evaluate one selected state so callers can checkpoint by state."""

        if provider.basis_identity != bundle.basis.basis_identity:
            raise SolverError("basis provider differs from solved AO layout")
        if isinstance(column, bool) or not 0 <= int(column) < (
            eigensystem.absolute_state_numbers.size
        ):
            raise SolverError("Orbital field column is out of range")
        normalized = int(column)
        state = int(eigensystem.absolute_state_numbers[normalized])
        field = provider.evaluate_orbital(
            bundle.structure,
            bundle.basis,
            eigensystem.eigenvectors[:, normalized],
            request.grid,
            resources,
            precision=request.precision,
        )
        return OrbitalStateField(
            state_number=state,
            label=_state_label(state, bundle),
            energy_ev=float(eigensystem.energies_ev[normalized]),
            field=field,
        )

    def close(self) -> None:
        close = getattr(self.solver, "close", None)
        if callable(close):
            close()


def _state_label(
    state_number: int,
    bundle: LocalizedOperatorBundle,
) -> str:
    filling = bundle.filling
    if filling is None:
        return f"state-{state_number}"
    vbm = filling.vbm_state_number
    cbm = filling.cbm_state_number
    if state_number == vbm:
        return "VBM"
    if state_number < vbm:
        return f"VBM-{vbm - state_number}"
    if state_number == cbm:
        return "CBM"
    return f"CBM+{state_number - cbm}"

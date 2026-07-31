"""Selected Gamma-point orbitals and resumable real-space fields."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from .data import (
    ElectronicFilling,
    LocalizedOperatorBundle,
    StateRange,
)
from .errors import InputError, SolverError
from .primme import IndexedResult, PrimmeCudssSolver
from .sparse import BlochPattern


_EDGE = re.compile(r"^(VBM|CBM)(?:([+-])(\d+))?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class GridSpec:
    shape: tuple[int, int, int] | None = None
    spacing_angstrom: float | None = None

    def __post_init__(self) -> None:
        if (self.shape is None) == (self.spacing_angstrom is None):
            raise InputError("grid needs either shape or spacing_angstrom")
        if self.shape is not None:
            shape = tuple(int(value) for value in self.shape)
            if len(shape) != 3 or any(value < 2 for value in shape):
                raise InputError("grid shape must contain three integers above one")
            object.__setattr__(self, "shape", shape)
        else:
            spacing = float(self.spacing_angstrom)
            if not math.isfinite(spacing) or spacing <= 0.0:
                raise InputError("grid spacing must be positive")
            object.__setattr__(self, "spacing_angstrom", spacing)

    def resolve_shape(
        self,
        cell_angstrom: np.ndarray,
    ) -> tuple[int, int, int]:
        if self.shape is not None:
            return self.shape
        assert self.spacing_angstrom is not None
        return tuple(
            max(2, int(math.ceil(length / self.spacing_angstrom)))
            for length in np.linalg.norm(cell_angstrom, axis=1)
        )


@dataclass(frozen=True, slots=True)
class FieldResources:
    device: str = "cuda:0"
    chunk_points: int = 65_536

    def __post_init__(self) -> None:
        if not str(self.device).strip() or int(self.chunk_points) < 1:
            raise InputError("invalid orbital field resources")
        object.__setattr__(self, "device", str(self.device).strip())
        object.__setattr__(self, "chunk_points", int(self.chunk_points))


@dataclass(frozen=True, slots=True)
class ResolvedSelection:
    expression: str
    states: StateRange
    semantics: str


@dataclass(slots=True)
class OrbitalEigensystem:
    selection: ResolvedSelection
    state_numbers: np.ndarray
    energies_ev: np.ndarray
    eigenvectors: np.ndarray
    phase_anchor_indices: np.ndarray
    maximum_residual_ev: float
    stage_seconds: dict[str, float]
    peak_gpu_memory_bytes: int | None
    warnings: tuple[str, ...]


@dataclass(slots=True)
class OrbitalField:
    values: np.ndarray
    cell_angstrom: np.ndarray
    origin_angstrom: np.ndarray
    voxel_volume_bohr3: float
    compute_device: str

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.complex64)
        self.cell_angstrom = np.asarray(self.cell_angstrom, dtype=np.float64)
        self.origin_angstrom = np.asarray(
            self.origin_angstrom,
            dtype=np.float64,
        )
        if (
            self.values.ndim != 3
            or self.cell_angstrom.shape != (3, 3)
            or self.origin_angstrom.shape != (3,)
            or self.voxel_volume_bohr3 <= 0.0
        ):
            raise InputError("invalid orbital field")

    @property
    def grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.values.shape)

    @property
    def captured_norm(self) -> float:
        return float(
            np.sum(np.abs(self.values) ** 2, dtype=np.float64)
            * self.voxel_volume_bohr3
        )


@dataclass(slots=True)
class OrbitalStateField:
    state_number: int
    label: str
    energy_ev: float
    field: OrbitalField
    evaluation_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ProbabilityLevels:
    probabilities: np.ndarray
    density_levels: np.ndarray
    amplitude_levels: np.ndarray
    captured_norm: float


def _edge_state(token: str, filling: ElectronicFilling) -> int:
    match = _EDGE.fullmatch(token.strip())
    if match is None:
        raise InputError(f"invalid band-edge state: {token}")
    edge, sign, magnitude = match.groups()
    base = (
        filling.vbm_state_number
        if edge.upper() == "VBM"
        else filling.cbm_state_number
    )
    offset = int(magnitude or 0)
    return base - offset if sign == "-" else base + offset


def resolve_state_selection(
    expression: str,
    filling: ElectronicFilling | None,
) -> ResolvedSelection:
    expression = expression.strip()
    if expression.lower().startswith("absolute:"):
        values = expression.split(":")[1:]
        if len(values) not in {1, 2}:
            raise InputError("absolute selection must be absolute:N or absolute:N:M")
        try:
            first = int(values[0])
            last = first if len(values) == 1 else int(values[1])
        except ValueError as exc:
            raise InputError("absolute state numbers must be integers") from exc
        return ResolvedSelection(expression, StateRange(first, last), "absolute")
    if filling is None:
        raise InputError("VBM/CBM state selection needs bundle filling")
    values = expression.split(":")
    if len(values) not in {1, 2}:
        raise InputError("state selection must be STATE or STATE:STATE")
    first = _edge_state(values[0], filling)
    last = first if len(values) == 1 else _edge_state(values[1], filling)
    return ResolvedSelection(
        expression,
        StateRange(first, last),
        "band_edge",
    )


def state_label(
    state_number: int,
    filling: ElectronicFilling | None,
) -> str:
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


def _phase_fix(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fixed = np.array(vectors, dtype=np.complex64, copy=True, order="F")
    anchors = np.argmax(np.abs(fixed), axis=0).astype(np.int64)
    for column, anchor in enumerate(anchors):
        value = fixed[int(anchor), column]
        if value == 0.0:
            raise SolverError("cannot phase-fix a zero eigenvector")
        fixed[:, column] *= np.exp(-1j * np.angle(value))
        if fixed[int(anchor), column].real < 0.0:
            fixed[:, column] *= -1.0
    return fixed, anchors


@dataclass(slots=True)
class OrbitalCalculation:
    bundle: LocalizedOperatorBundle
    states: str
    grid: GridSpec
    energy_hint_ev: float | None = None
    solver: Any | None = None
    selection: ResolvedSelection = field(init=False)

    def __post_init__(self) -> None:
        self.selection = resolve_state_selection(
            self.states,
            self.bundle.filling,
        )
        if self.solver is None:
            self.solver = PrimmeCudssSolver()

    def solve_eigensystem(self) -> OrbitalEigensystem:
        pattern = BlochPattern.from_bundle(self.bundle)
        matrices = pattern.workspace(complex_values=False).update(
            self.bundle,
            (0.0, 0.0, 0.0),
        )
        hint = self.energy_hint_ev
        if (
            hint is None
            and self.bundle.filling is not None
            and self.bundle.filling.fermi_energy_ev is not None
        ):
            hint = self.bundle.filling.fermi_energy_ev
        solved: IndexedResult = self.solver.solve_states(
            matrices,
            self.selection.states,
            energy_hint_ev=0.0 if hint is None else float(hint),
        )
        fixed, anchors = _phase_fix(solved.eigenvectors)
        warnings = list(solved.warnings)
        if self.bundle.basis.spinor_width != 1:
            warnings.append(
                "spinor input is outside the current non-SOC Orbital scope; "
                "the result may be inaccurate"
            )
        if not all(self.bundle.structure.periodic):
            warnings.append(
                "non-periodic axes are outside the current Orbital scope; "
                "the result may be inaccurate"
            )
        for index, gap in enumerate(np.diff(solved.energies_ev)):
            if abs(float(gap)) <= 1.0e-5:
                left = int(solved.state_numbers[index])
                right = int(solved.state_numbers[index + 1])
                warnings.append(
                    f"states {left} and {right} are near-degenerate; "
                    "individual orbital shapes may depend on the solver basis"
                )
        return OrbitalEigensystem(
            selection=self.selection,
            state_numbers=solved.state_numbers,
            energies_ev=solved.energies_ev,
            eigenvectors=fixed,
            phase_anchor_indices=anchors,
            maximum_residual_ev=solved.maximum_residual_ev,
            stage_seconds=solved.stage_seconds,
            peak_gpu_memory_bytes=solved.peak_gpu_memory_bytes,
            warnings=tuple(warnings),
        )

    def evaluate_state(
        self,
        eigensystem: OrbitalEigensystem,
        provider: Any,
        resources: FieldResources,
        column: int,
    ) -> OrbitalStateField:
        if provider.basis_identity != self.bundle.basis.basis_identity:
            raise InputError("OpenMX basis differs from the solved AO layout")
        state = int(eigensystem.state_numbers[column])
        field = provider.evaluate_orbital(
            self.bundle.structure,
            self.bundle.basis,
            eigensystem.eigenvectors[:, column],
            self.grid,
            resources,
        )
        return OrbitalStateField(
            state,
            state_label(state, self.bundle.filling),
            float(eigensystem.energies_ev[column]),
            field,
        )

    def close(self) -> None:
        close = getattr(self.solver, "close", None)
        if callable(close):
            close()


def enclosed_probability_levels(
    field: OrbitalField,
) -> ProbabilityLevels:
    probabilities = np.linspace(0.50, 0.99, 50, dtype=np.float64)
    density = np.asarray(np.abs(field.values) ** 2, dtype=np.float64).reshape(-1)
    total = float(np.sum(density) * field.voxel_volume_bohr3)
    if total <= 0.0 or not math.isfinite(total):
        raise SolverError("orbital field has no positive captured norm")
    descending = np.sort(density)[::-1]
    cumulative = (
        np.cumsum(descending) * field.voxel_volume_bohr3 / total
    )
    positions = np.searchsorted(cumulative, probabilities, side="left")
    levels = descending[np.clip(positions, 0, descending.size - 1)]
    return ProbabilityLevels(
        probabilities,
        levels,
        np.sqrt(levels),
        total,
    )


def write_eigensystem(
    path: str | Path,
    eigensystem: OrbitalEigensystem,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            expression=np.asarray(eigensystem.selection.expression),
            semantics=np.asarray(eigensystem.selection.semantics),
            first=np.asarray(eigensystem.selection.states.first),
            last=np.asarray(eigensystem.selection.states.last),
            state_numbers=eigensystem.state_numbers,
            energies_ev=eigensystem.energies_ev,
            eigenvectors=eigensystem.eigenvectors,
            phase_anchor_indices=eigensystem.phase_anchor_indices,
            maximum_residual_ev=np.asarray(
                eigensystem.maximum_residual_ev,
                dtype=np.float64,
            ),
            stage_names=np.asarray(tuple(eigensystem.stage_seconds), dtype="U64"),
            stage_values=np.asarray(
                tuple(eigensystem.stage_seconds.values()),
                dtype=np.float64,
            ),
            peak_gpu_memory_bytes=np.asarray(
                -1
                if eigensystem.peak_gpu_memory_bytes is None
                else eigensystem.peak_gpu_memory_bytes,
                dtype=np.int64,
            ),
            warnings=np.asarray(eigensystem.warnings, dtype="U256"),
        )
    os.replace(temporary, path)


def read_eigensystem(path: str | Path) -> OrbitalEigensystem | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            peak = int(values["peak_gpu_memory_bytes"])
            selection = ResolvedSelection(
                str(values["expression"].item()),
                StateRange(int(values["first"]), int(values["last"])),
                str(values["semantics"].item()),
            )
            return OrbitalEigensystem(
                selection=selection,
                state_numbers=values["state_numbers"],
                energies_ev=values["energies_ev"],
                eigenvectors=values["eigenvectors"],
                phase_anchor_indices=values["phase_anchor_indices"],
                maximum_residual_ev=float(values["maximum_residual_ev"]),
                stage_seconds=dict(
                    zip(
                        values["stage_names"].tolist(),
                        values["stage_values"].tolist(),
                    )
                ),
                peak_gpu_memory_bytes=None if peak < 0 else peak,
                warnings=tuple(values["warnings"].tolist()),
            )
    except (OSError, ValueError, KeyError) as exc:
        raise InputError(f"cannot read Orbital eigensystem {path}: {exc}") from exc


def write_state_field(path: str | Path, state: OrbitalStateField) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            state_number=np.asarray(state.state_number),
            label=np.asarray(state.label),
            energy_ev=np.asarray(state.energy_ev),
            values=state.field.values,
            cell_angstrom=state.field.cell_angstrom,
            origin_angstrom=state.field.origin_angstrom,
            voxel_volume_bohr3=np.asarray(state.field.voxel_volume_bohr3),
            compute_device=np.asarray(state.field.compute_device),
            evaluation_seconds=np.asarray(
                state.evaluation_seconds,
                dtype=np.float64,
            ),
        )
    os.replace(temporary, path)


def read_state_field(path: str | Path) -> OrbitalStateField | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            return OrbitalStateField(
                int(values["state_number"]),
                str(values["label"].item()),
                float(values["energy_ev"]),
                OrbitalField(
                    values["values"],
                    values["cell_angstrom"],
                    values["origin_angstrom"],
                    float(values["voxel_volume_bohr3"]),
                    str(values["compute_device"].item()),
                ),
                float(values["evaluation_seconds"]),
            )
    except (OSError, ValueError, KeyError) as exc:
        raise InputError(f"cannot read Orbital field {path}: {exc}") from exc

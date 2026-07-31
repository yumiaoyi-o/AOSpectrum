"""Energy-window Band calculation, lightweight resume, and plotting."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .ciss import CissSettings, CudssCissSolver
from .data import EnergyInterval, LocalizedOperatorBundle
from .errors import InputError, SolverError
from .sparse import BlochPattern


CONTINUITY_GUARD_EV = 0.02
KPATH_SCHEMA = "aospectrum.kpath/v1"


@dataclass(frozen=True, slots=True)
class KPath:
    fractional_points: np.ndarray
    node_indices: np.ndarray
    node_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        points = np.asarray(self.fractional_points, dtype=np.float64)
        nodes = np.asarray(self.node_indices, dtype=np.int64)
        labels = tuple(str(value) for value in self.node_labels)
        if points.ndim != 2 or points.shape[1:] != (3,) or points.shape[0] < 2:
            raise InputError("k path must contain at least two 3D points")
        if (
            nodes.ndim != 1
            or nodes.size < 2
            or nodes[0] != 0
            or nodes[-1] != points.shape[0] - 1
            or np.any(np.diff(nodes) <= 0)
            or len(labels) != nodes.size
        ):
            raise InputError("invalid k-path nodes or labels")
        points.setflags(write=False)
        nodes.setflags(write=False)
        object.__setattr__(self, "fractional_points", points)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "node_labels", labels)

    @classmethod
    def interpolate(
        cls,
        nodes: Sequence[Sequence[float]],
        labels: Sequence[str],
        *,
        points_per_segment: int,
    ) -> "KPath":
        coordinates = np.asarray(nodes, dtype=np.float64)
        count = int(points_per_segment)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1:] != (3,)
            or coordinates.shape[0] < 2
            or count < 1
        ):
            raise InputError("invalid interpolated k path")
        parts: list[np.ndarray] = []
        node_indices = [0]
        for index in range(coordinates.shape[0] - 1):
            segment = np.linspace(
                coordinates[index],
                coordinates[index + 1],
                count + 1,
                dtype=np.float64,
            )
            parts.append(segment if index == 0 else segment[1:])
            node_indices.append(node_indices[-1] + count)
        return cls(
            np.concatenate(parts),
            np.asarray(node_indices),
            tuple(labels),
        )

    @property
    def n_points(self) -> int:
        return int(self.fractional_points.shape[0])

    def distances(self, cell_angstrom: np.ndarray) -> np.ndarray:
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell_angstrom).T
        cartesian = self.fractional_points @ reciprocal
        distances = np.zeros(self.n_points, dtype=np.float64)
        distances[1:] = np.cumsum(
            np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        )
        return distances


def read_kpath(path: str | Path) -> KPath:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read k path {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError("k path must contain a JSON object")
    if value.get("schema", KPATH_SCHEMA) != KPATH_SCHEMA:
        raise InputError(f"k path schema must be {KPATH_SCHEMA}")
    labels = value.get("labels")
    if not isinstance(labels, list):
        raise InputError("k path labels must be a list")
    if "nodes_fractional" in value:
        return KPath.interpolate(
            value["nodes_fractional"],
            labels,
            points_per_segment=value.get("points_per_segment", 0),
        )
    try:
        return KPath(
            value["fractional_points"],
            value["node_indices"],
            tuple(labels),
        )
    except KeyError as exc:
        raise InputError(
            "k path needs nodes_fractional or fractional_points"
        ) from exc


def resolve_energy_reference(
    bundle: LocalizedOperatorBundle,
    mode: str,
    value_ev: float | None = None,
) -> float:
    if mode in {"input_zero", "absolute"}:
        return 0.0
    if mode == "fixed":
        if value_ev is None:
            raise InputError("fixed energy reference requires a value")
        return float(value_ev)
    if mode == "bundle_fermi":
        if bundle.filling is None or bundle.filling.fermi_energy_ev is None:
            raise InputError("bundle has no Fermi energy")
        return bundle.filling.fermi_energy_ev
    raise InputError("energy reference must be input_zero, fixed, or bundle_fermi")


@dataclass(slots=True)
class BandPoint:
    index: int
    k_fractional: np.ndarray
    energies_ev: np.ndarray
    maximum_residual_ev: float
    stage_seconds: dict[str, float]
    assembly_seconds: float
    peak_gpu_memory_bytes: int | None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.index = int(self.index)
        self.k_fractional = np.asarray(self.k_fractional, dtype=np.float64)
        self.energies_ev = np.asarray(self.energies_ev, dtype=np.float32)
        if self.index < 0 or self.k_fractional.shape != (3,):
            raise InputError("invalid Band point")


@dataclass(slots=True)
class BandResult:
    kpath: KPath
    display_interval_ev: EnergyInterval
    solve_interval_ev: EnergyInterval
    reference_energy_ev: float
    cell_angstrom: np.ndarray
    points: tuple[BandPoint, ...]
    elapsed_seconds: float
    warnings: tuple[str, ...] = ()

    @property
    def display_energies_ev(self) -> tuple[np.ndarray, ...]:
        return tuple(
            point.energies_ev - self.reference_energy_ev
            for point in self.points
        )


@dataclass(slots=True)
class BandCalculation:
    bundle: LocalizedOperatorBundle
    kpath: KPath
    display_interval_ev: EnergyInterval
    reference_energy_ev: float
    settings: CissSettings = field(default_factory=CissSettings)
    solver: Any | None = None
    solve_interval_ev: EnergyInterval = field(init=False)
    absolute_interval_ev: EnergyInterval = field(init=False)
    pattern: BlochPattern = field(init=False)
    workspace: Any = field(init=False)
    warnings: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.solve_interval_ev = self.display_interval_ev.expanded(
            CONTINUITY_GUARD_EV
        )
        self.absolute_interval_ev = EnergyInterval(
            self.solve_interval_ev.lower_ev + self.reference_energy_ev,
            self.solve_interval_ev.upper_ev + self.reference_energy_ev,
        )
        self.pattern = BlochPattern.from_bundle(self.bundle)
        self.workspace = self.pattern.workspace(complex_values=True)
        if self.solver is None:
            self.solver = CudssCissSolver(self.settings)
        warnings: list[str] = []
        if self.bundle.basis.spinor_width != 1:
            warnings.append(
                "spinor input is outside the current non-SOC Band scope; "
                "the result may be inaccurate"
            )
        if not all(self.bundle.structure.periodic):
            warnings.append(
                "non-periodic axes are outside the current Bloch scope; "
                "the result may be inaccurate"
            )
        self.warnings = tuple(warnings)

    def solve_point(self, index: int) -> BandPoint:
        if not 0 <= int(index) < self.kpath.n_points:
            raise InputError("k-point index is out of range")
        index = int(index)
        kpoint = self.kpath.fractional_points[index]
        started = perf_counter()
        matrices = self.workspace.update(self.bundle, kpoint)
        assembly_seconds = perf_counter() - started
        result = self.solver.solve_interval(
            matrices,
            self.absolute_interval_ev,
        )
        if not result.complete:
            raise SolverError(
                "CISS search subspace is saturated; increase block_size "
                "or moment_size"
            )
        return BandPoint(
            index=index,
            k_fractional=kpoint,
            energies_ev=result.energies_ev,
            maximum_residual_ev=result.maximum_residual_ev,
            stage_seconds=result.stage_seconds,
            assembly_seconds=assembly_seconds,
            peak_gpu_memory_bytes=result.peak_gpu_memory_bytes,
            warnings=result.warnings,
        )

    def build_result(
        self,
        points: Sequence[BandPoint],
        *,
        elapsed_seconds: float,
    ) -> BandResult:
        ordered = tuple(sorted(points, key=lambda point: point.index))
        if [point.index for point in ordered] != list(range(self.kpath.n_points)):
            raise SolverError("Band result does not contain every k point")
        return BandResult(
            kpath=self.kpath,
            display_interval_ev=self.display_interval_ev,
            solve_interval_ev=self.solve_interval_ev,
            reference_energy_ev=self.reference_energy_ev,
            cell_angstrom=self.bundle.structure.cell_angstrom,
            points=ordered,
            elapsed_seconds=float(elapsed_seconds),
            warnings=self.warnings,
        )

    def close(self) -> None:
        close = getattr(self.solver, "close", None)
        if callable(close):
            close()


def write_point(path: str | Path, point: BandPoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            index=np.asarray(point.index, dtype=np.int64),
            k_fractional=point.k_fractional,
            energies_ev=point.energies_ev,
            maximum_residual_ev=np.asarray(
                point.maximum_residual_ev,
                dtype=np.float64,
            ),
            stage_names=np.asarray(tuple(point.stage_seconds), dtype="U64"),
            stage_values=np.asarray(
                tuple(point.stage_seconds.values()),
                dtype=np.float64,
            ),
            assembly_seconds=np.asarray(point.assembly_seconds, dtype=np.float64),
            peak_gpu_memory_bytes=np.asarray(
                -1
                if point.peak_gpu_memory_bytes is None
                else point.peak_gpu_memory_bytes,
                dtype=np.int64,
            ),
            warnings=np.asarray(point.warnings, dtype="U256"),
        )
    os.replace(temporary, path)


def read_point(
    path: str | Path,
    *,
    index: int,
    k_fractional: np.ndarray,
) -> BandPoint | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            observed_index = int(values["index"])
            observed_kpoint = np.asarray(values["k_fractional"])
            if observed_index != index or not np.array_equal(
                observed_kpoint,
                k_fractional,
            ):
                raise InputError(f"Band checkpoint does not match point {index}")
            names = values["stage_names"].tolist()
            seconds = values["stage_values"].tolist()
            peak = int(values["peak_gpu_memory_bytes"])
            return BandPoint(
                index=index,
                k_fractional=observed_kpoint,
                energies_ev=values["energies_ev"],
                maximum_residual_ev=float(values["maximum_residual_ev"]),
                stage_seconds=dict(zip(names, seconds)),
                assembly_seconds=float(values["assembly_seconds"]),
                peak_gpu_memory_bytes=None if peak < 0 else peak,
                warnings=tuple(values["warnings"].tolist()),
            )
    except (OSError, ValueError, KeyError) as exc:
        raise InputError(f"cannot read Band checkpoint {path}: {exc}") from exc


def assign_track_ids(energies: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    first = np.arange(energies[0].size, dtype=np.int64)
    tracks = [first]
    next_identifier = int(first.size)
    for previous, current in zip(energies, energies[1:]):
        previous_ids = tracks[-1]
        current_ids = np.full(current.size, -1, dtype=np.int64)
        if previous.size and current.size:
            rows, columns = linear_sum_assignment(
                np.abs(previous[:, None] - current[None, :])
            )
            current_ids[columns] = previous_ids[rows]
        unmatched = np.flatnonzero(current_ids < 0)
        current_ids[unmatched] = np.arange(
            next_identifier,
            next_identifier + unmatched.size,
        )
        next_identifier += int(unmatched.size)
        tracks.append(current_ids)
    return tuple(tracks)


def _line_segments(
    x: np.ndarray,
    energies: Sequence[np.ndarray],
    tracks: Sequence[np.ndarray],
) -> np.ndarray:
    segments: list[np.ndarray] = []
    for index in range(x.size - 1):
        right = {
            int(track): local
            for local, track in enumerate(tracks[index + 1])
        }
        for local, track in enumerate(tracks[index]):
            match = right.get(int(track))
            if match is not None:
                segments.append(
                    np.asarray(
                        [
                            [x[index], energies[index][local]],
                            [x[index + 1], energies[index + 1][match]],
                        ]
                    )
                )
    if not segments:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack(segments)


def write_band_output(result: BandResult, root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    distances = result.kpath.distances(result.cell_angstrom)
    energies = result.display_energies_ev
    tracks = assign_track_ids(energies)

    with (root / "bands.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "k_index",
                "k_distance_inv_angstrom",
                "track_id",
                "energy_ev",
            )
        )
        for point, distance, identifiers, values in zip(
            result.points,
            distances,
            tracks,
            energies,
        ):
            for track, energy in zip(identifiers, values):
                writer.writerow(
                    (
                        point.index,
                        f"{distance:.12e}",
                        int(track),
                        f"{float(energy):.12e}",
                    )
                )

    stage_seconds: dict[str, float] = {}
    warnings = list(result.warnings)
    for point in result.points:
        for name, seconds in point.stage_seconds.items():
            stage_seconds[name] = stage_seconds.get(name, 0.0) + float(seconds)
        warnings.extend(point.warnings)
    peak_values = [
        point.peak_gpu_memory_bytes
        for point in result.points
        if point.peak_gpu_memory_bytes is not None
    ]
    summary = {
        "backend": "ciss-cudss",
        "scalar_dtype": "complex64",
        "kpoint_count": result.kpath.n_points,
        "energy_interval_ev": [
            result.display_interval_ev.lower_ev,
            result.display_interval_ev.upper_ev,
        ],
        "reference_energy_ev": result.reference_energy_ev,
        "maximum_eigenpair_residual_ev": max(
            (point.maximum_residual_ev for point in result.points),
            default=0.0,
        ),
        "elapsed_seconds": result.elapsed_seconds,
        "assembly_seconds": sum(
            point.assembly_seconds for point in result.points
        ),
        "solver_stage_seconds": stage_seconds,
        "peak_gpu_memory_bytes": max(peak_values) if peak_values else None,
        "warnings": list(dict.fromkeys(warnings)),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    segments = _line_segments(distances, energies, tracks)
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    if segments.size:
        axis.add_collection(
            LineCollection(
                segments,
                colors="#17213a",
                linewidths=1.05,
                antialiased=True,
            )
        )
    axis.set_xlim(float(distances[0]), float(distances[-1]))
    axis.set_ylim(
        result.display_interval_ev.lower_ev,
        result.display_interval_ev.upper_ev,
    )
    node_x = distances[result.kpath.node_indices]
    for position in node_x[1:-1]:
        axis.axvline(position, color="#9aa1ad", linewidth=0.7, zorder=0)
    axis.axhline(0.0, color="#b73b3b", linewidth=0.8, linestyle="--")
    axis.set_xticks(node_x, result.kpath.node_labels)
    axis.set_xlabel("K path")
    axis.set_ylabel("Energy relative to reference (eV)")
    axis.tick_params(direction="in")
    figure.savefig(root / "band.png", dpi=240)
    plt.close(figure)
    return root

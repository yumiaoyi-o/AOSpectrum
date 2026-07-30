"""Transparent directory I/O for Band results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from aospectrum.band.model import BandPoint, BandResult, KPath
from aospectrum.band.tracking import assign_track_ids
from aospectrum.errors import BundleIOError
from aospectrum.model.spectra import (
    EnergyInterval,
    NumericalQuality,
    QualityPolicy,
    SolverReceipt,
)


SCHEMA = "aospectrum.band-result/v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_value(value),
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_band_result(result: BandResult, root: str | Path) -> Path:
    destination = Path(root)
    if destination.exists():
        if not destination.is_dir():
            raise BundleIOError(
                f"Band destination is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise BundleIOError(
                f"Band destination is not empty: {destination}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(
        [point.absolute_energies_ev.size for point in result.points],
        dtype=np.int64,
    )
    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    arrays = {
        "kpoints.npy": result.kpath.fractional_points,
        "node_indices.npy": result.kpath.node_indices,
        "energy_offsets.npy": offsets,
        "absolute_energies_ev.npy": np.concatenate(
            [point.absolute_energies_ev for point in result.points]
        ),
        "raw_residuals_ev.npy": np.concatenate(
            [point.quality.raw_residuals_ev for point in result.points]
        ),
        "normalized_residuals.npy": np.concatenate(
            [point.quality.normalized_residuals for point in result.points]
        ),
    }
    for name, values in arrays.items():
        np.save(destination / name, values, allow_pickle=False)
    _write_band_table(result, destination / "bands.csv")
    manifest = {
        "schema": SCHEMA,
        "precision": result.precision,
        "backend": result.backend,
        "display_interval_ev": [
            result.display_interval_ev.lower_ev,
            result.display_interval_ev.upper_ev,
        ],
        "solve_interval_ev": [
            result.solve_interval_ev.lower_ev,
            result.solve_interval_ev.upper_ev,
        ],
        "reference_energy_ev": result.reference_energy_ev,
        "node_labels": list(result.kpath.node_labels),
        "quality_status": result.quality_status,
        "metadata": dict(result.metadata),
        "points": [
            {
                "index": point.index,
                "quality_policy": {
                    "name": point.quality.policy.name,
                    "raw_residual_tolerance_ev": (
                        point.quality.policy.raw_residual_tolerance_ev
                    ),
                    "normalized_residual_tolerance": (
                        point.quality.policy.normalized_residual_tolerance
                    ),
                    "s_orthogonality_tolerance": (
                        point.quality.policy.s_orthogonality_tolerance
                    ),
                },
                "s_orthogonality_error": point.quality.s_orthogonality_error,
                "calculation_dtype": point.quality.calculation_dtype,
                "receipt": {
                    "backend": point.receipt.backend,
                    "device": point.receipt.device,
                    "scalar_dtype": point.receipt.scalar_dtype,
                    "stage_seconds": dict(point.receipt.stage_seconds),
                    "peak_host_rss_bytes": point.receipt.peak_host_rss_bytes,
                    "peak_gpu_memory_bytes": point.receipt.peak_gpu_memory_bytes,
                    "counters": dict(point.receipt.counters),
                    "details": dict(point.receipt.details),
                },
                "assembly": dict(point.assembly_metadata),
            }
            for point in result.points
        ],
    }
    _write_json(destination / "manifest.json", manifest)
    _write_json(
        destination / "quality.json",
        {
            "schema": "aospectrum.band-quality/v1",
            "status": result.quality_status,
            "points": [
                {
                    "index": point.index,
                    "status": point.quality.status,
                    "warnings": list(point.quality.warnings),
                    "maximum_raw_residual_ev": float(
                        np.max(point.quality.raw_residuals_ev, initial=0.0)
                    ),
                    "maximum_normalized_residual": float(
                        np.max(point.quality.normalized_residuals, initial=0.0)
                    ),
                    "s_orthogonality_error": (
                        point.quality.s_orthogonality_error
                    ),
                }
                for point in result.points
            ],
        },
    )
    _write_json(
        destination / "receipt.json",
        {
            "schema": "aospectrum.band-receipt/v1",
            "backend": result.backend,
            "precision": result.precision,
            "points": [
                {
                    "index": point.index,
                    "receipt": manifest["points"][point.index]["receipt"],
                    "assembly": manifest["points"][point.index]["assembly"],
                }
                for point in result.points
            ],
        },
    )
    return destination


def _write_band_table(result: BandResult, path: Path) -> None:
    distances = result.kpath.distances(
        np.asarray(
            result.metadata.get("cell_angstrom", np.eye(3)),
            dtype=np.float64,
        )
    )
    tracks = assign_track_ids(result)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "k_index",
                "k_distance_inv_angstrom",
                "local_order",
                "track_id",
                "energy_raw_ev",
                "energy_plot_ev",
                "raw_residual_ev",
                "normalized_residual",
            )
        )
        for point, distance, identifiers in zip(
            result.points,
            distances,
            tracks,
        ):
            for local, (energy, track, raw, normalized) in enumerate(
                zip(
                    point.absolute_energies_ev,
                    identifiers,
                    point.quality.raw_residuals_ev,
                    point.quality.normalized_residuals,
                )
            ):
                writer.writerow(
                    (
                        point.index,
                        f"{float(distance):.16e}",
                        local,
                        int(track),
                        f"{float(energy):.16e}",
                        f"{float(energy - result.reference_energy_ev):.16e}",
                        f"{float(raw):.16e}",
                        f"{float(normalized):.16e}",
                    )
                )


def read_band_result(root: str | Path) -> BandResult:
    source = Path(root)
    try:
        manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema") != SCHEMA:
            raise BundleIOError("unsupported Band result schema")
        kpoints = np.load(source / "kpoints.npy", mmap_mode="r")
        nodes = np.load(source / "node_indices.npy", mmap_mode="r")
        offsets = np.load(source / "energy_offsets.npy", mmap_mode="r")
        energies = np.load(source / "absolute_energies_ev.npy", mmap_mode="r")
        raw = np.load(source / "raw_residuals_ev.npy", mmap_mode="r")
        normalized = np.load(
            source / "normalized_residuals.npy",
            mmap_mode="r",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BundleIOError(f"cannot read Band result {source}: {exc}") from exc
    if kpoints.ndim != 2 or kpoints.shape[1:] != (3,):
        raise BundleIOError("Band kpoints.npy must have shape (n_points, 3)")
    if nodes.ndim != 1:
        raise BundleIOError("Band node_indices.npy must be one-dimensional")
    if (
        offsets.ndim != 1
        or offsets.size != kpoints.shape[0] + 1
        or offsets.dtype.kind not in "iu"
        or int(offsets[0]) != 0
        or np.any(np.diff(offsets) < 0)
    ):
        raise BundleIOError("Band energy offsets are invalid")
    array_lengths = {
        int(np.asarray(values).size)
        for values in (energies, raw, normalized)
        if np.asarray(values).ndim == 1
    }
    if (
        energies.ndim != 1
        or raw.ndim != 1
        or normalized.ndim != 1
        or len(array_lengths) != 1
        or int(offsets[-1]) not in array_lengths
    ):
        raise BundleIOError("Band energy and residual arrays do not match offsets")
    records = manifest.get("points")
    if not isinstance(records, list) or len(records) != kpoints.shape[0]:
        raise BundleIOError("Band point records do not match k points")
    points: list[BandPoint] = []
    for index, record in enumerate(records):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        policy_record = record["quality_policy"]
        policy = QualityPolicy(**policy_record)
        receipt_record = record["receipt"]
        receipt = SolverReceipt(
            backend=receipt_record["backend"],
            device=receipt_record["device"],
            scalar_dtype=receipt_record["scalar_dtype"],
            stage_seconds=receipt_record["stage_seconds"],
            peak_host_rss_bytes=receipt_record["peak_host_rss_bytes"],
            peak_gpu_memory_bytes=receipt_record["peak_gpu_memory_bytes"],
            counters=receipt_record["counters"],
            details=receipt_record["details"],
        )
        points.append(
            BandPoint(
                index=index,
                k_fractional=kpoints[index],
                absolute_energies_ev=energies[start:stop],
                quality=NumericalQuality(
                    raw_residuals_ev=raw[start:stop],
                    normalized_residuals=normalized[start:stop],
                    s_orthogonality_error=record["s_orthogonality_error"],
                    calculation_dtype=record["calculation_dtype"],
                    policy=policy,
                ),
                receipt=receipt,
                assembly_metadata=record["assembly"],
            )
        )
    return BandResult(
        kpath=KPath(kpoints, nodes, tuple(manifest["node_labels"])),
        display_interval_ev=EnergyInterval(*manifest["display_interval_ev"]),
        solve_interval_ev=EnergyInterval(*manifest["solve_interval_ev"]),
        reference_energy_ev=float(manifest["reference_energy_ev"]),
        points=tuple(points),
        precision=manifest["precision"],
        backend=manifest["backend"],
        metadata=manifest["metadata"],
    )

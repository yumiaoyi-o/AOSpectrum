"""Finite per-k-point checkpoints and deterministic Band sharding."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import numpy as np

from aospectrum import __version__
from aospectrum.band.calculator import BandCalculator, PreparedBandCalculation
from aospectrum.band.model import BandPoint, BandRequest, BandResult
from aospectrum.errors import ArtifactError, ContractError
from aospectrum.model.operators import LocalizedOperatorBundle
from aospectrum.model.spectra import NumericalQuality, QualityPolicy, SolverReceipt


RUN_SCHEMA = "aospectrum.band-journal/v1"
POINT_SCHEMA = "aospectrum.band-point/v1"


def shard_indices(point_count: int, rank: int, world_size: int) -> tuple[int, ...]:
    if (
        isinstance(point_count, bool)
        or isinstance(rank, bool)
        or isinstance(world_size, bool)
        or int(point_count) <= 0
        or int(world_size) <= 0
        or not 0 <= int(rank) < int(world_size)
    ):
        raise ContractError("Band shard dimensions are invalid")
    return tuple(range(int(rank), int(point_count), int(world_size)))


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def band_run_specification(
    bundle: LocalizedOperatorBundle,
    request: BandRequest,
    backend: object,
    *,
    world_size: int = 1,
) -> dict[str, Any]:
    bundle_identity = bundle.provenance.get("bundle_manifest_sha256")
    if bundle_identity is None:
        bundle_identity = (
            f"in-memory:{bundle.basis.basis_identity}:"
            f"{bundle.n_orbitals}:{bundle.operators.n_blocks}"
        )
    backend_name = getattr(backend, "name", None)
    if not isinstance(backend_name, str) or not backend_name:
        raise ArtifactError("Band backend has no stable name")
    settings = getattr(backend, "settings", None)
    backend_settings = (
        asdict(settings)
        if settings is not None and is_dataclass(settings)
        else None
    )
    return {
        "schema": RUN_SCHEMA,
        "aospectrum_version": __version__,
        "world_size": int(world_size),
        "bundle_identity": str(bundle_identity),
        "basis_identity": bundle.basis.basis_identity,
        "backend": {
            "name": backend_name,
            "settings": _json_value(backend_settings),
        },
        "precision": request.precision,
        "display_interval_ev": [
            request.display_interval_ev.lower_ev,
            request.display_interval_ev.upper_ev,
        ],
        "energy_reference": {
            "mode": request.energy_reference.mode,
            "value_ev": request.energy_reference.value_ev,
        },
        "kpath": {
            "fractional_points": request.kpath.fractional_points.tolist(),
            "node_indices": request.kpath.node_indices.tolist(),
            "node_labels": list(request.kpath.node_labels),
        },
    }


class BandPointJournal:
    """One immutable run specification plus committed point records."""

    def __init__(self, root: str | Path, specification: Mapping[str, Any]) -> None:
        self.root = Path(root)
        self.points_root = self.root / "points"
        normalized = _json_value(dict(specification))
        if normalized.get("schema") != RUN_SCHEMA:
            raise ArtifactError("Band journal specification schema is invalid")
        if self.root.exists():
            try:
                observed = json.loads(
                    (self.root / "run.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(
                    f"cannot read Band journal {self.root}"
                ) from exc
            if _canonical_json(observed) != _canonical_json(normalized):
                raise ArtifactError("Band journal belongs to another request")
        else:
            self.points_root.mkdir(parents=True)
            temporary = self.root / f".run-{uuid4().hex}.json"
            temporary.write_text(
                json.dumps(normalized, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.root / "run.json")
        self.points_root.mkdir(exist_ok=True)
        self.specification = normalized

    def _paths(self, index: int) -> tuple[Path, Path]:
        stem = f"k-{int(index):06d}"
        return self.points_root / f"{stem}.npz", self.points_root / f"{stem}.json"

    def load_point(
        self,
        index: int,
        expected_kpoint: np.ndarray,
    ) -> BandPoint | None:
        arrays_path, marker_path = self._paths(index)
        if not marker_path.exists():
            return None
        if not arrays_path.is_file():
            raise ArtifactError(f"Band marker has no arrays: {marker_path}")
        try:
            record = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                record.get("schema") != POINT_SCHEMA
                or int(record.get("index", -1)) != int(index)
                or _sha256(arrays_path) != record.get("arrays_sha256")
            ):
                raise ArtifactError(f"Band point checkpoint is invalid: {marker_path}")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                energies = np.array(arrays["energies_ev"], copy=True)
                raw = np.array(arrays["raw_residuals_ev"], copy=True)
                normalized = np.array(
                    arrays["normalized_residuals"],
                    copy=True,
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"cannot decode Band point checkpoint {marker_path}"
            ) from exc
        kpoint = np.asarray(record["k_fractional"], dtype=np.float64)
        if not np.array_equal(kpoint, expected_kpoint):
            raise ArtifactError("Band checkpoint k point differs from request")
        policy = QualityPolicy(**record["quality"]["policy"])
        receipt_record = record["receipt"]
        return BandPoint(
            index=index,
            k_fractional=kpoint,
            absolute_energies_ev=energies,
            quality=NumericalQuality(
                raw_residuals_ev=raw,
                normalized_residuals=normalized,
                s_orthogonality_error=record["quality"][
                    "s_orthogonality_error"
                ],
                calculation_dtype=record["quality"]["calculation_dtype"],
                policy=policy,
            ),
            receipt=SolverReceipt(
                backend=receipt_record["backend"],
                device=receipt_record["device"],
                scalar_dtype=receipt_record["scalar_dtype"],
                stage_seconds=receipt_record["stage_seconds"],
                peak_host_rss_bytes=receipt_record["peak_host_rss_bytes"],
                peak_gpu_memory_bytes=receipt_record["peak_gpu_memory_bytes"],
                counters=receipt_record["counters"],
                details=receipt_record["details"],
            ),
            assembly_metadata=record["assembly"],
        )

    def write_point(self, point: BandPoint) -> None:
        arrays_path, marker_path = self._paths(point.index)
        existing = self.load_point(point.index, point.k_fractional)
        if existing is not None:
            return
        arrays_temp = self.points_root / f".{arrays_path.name}-{uuid4().hex}"
        marker_temp = self.points_root / f".{marker_path.name}-{uuid4().hex}"
        try:
            with arrays_temp.open("wb") as handle:
                np.savez(
                    handle,
                    energies_ev=point.absolute_energies_ev,
                    raw_residuals_ev=point.quality.raw_residuals_ev,
                    normalized_residuals=point.quality.normalized_residuals,
                )
            policy = point.quality.policy
            receipt = point.receipt
            record = {
                "schema": POINT_SCHEMA,
                "index": point.index,
                "k_fractional": point.k_fractional.tolist(),
                "arrays_sha256": _sha256(arrays_temp),
                "quality": {
                    "status": point.quality.status,
                    "warnings": list(point.quality.warnings),
                    "s_orthogonality_error": (
                        point.quality.s_orthogonality_error
                    ),
                    "calculation_dtype": point.quality.calculation_dtype,
                    "policy": {
                        "name": policy.name,
                        "raw_residual_tolerance_ev": (
                            policy.raw_residual_tolerance_ev
                        ),
                        "normalized_residual_tolerance": (
                            policy.normalized_residual_tolerance
                        ),
                        "s_orthogonality_tolerance": (
                            policy.s_orthogonality_tolerance
                        ),
                    },
                },
                "receipt": {
                    "backend": receipt.backend,
                    "device": receipt.device,
                    "scalar_dtype": receipt.scalar_dtype,
                    "stage_seconds": dict(receipt.stage_seconds),
                    "peak_host_rss_bytes": receipt.peak_host_rss_bytes,
                    "peak_gpu_memory_bytes": receipt.peak_gpu_memory_bytes,
                    "counters": dict(receipt.counters),
                    "details": dict(receipt.details),
                },
                "assembly": dict(point.assembly_metadata),
            }
            marker_temp.write_text(
                json.dumps(
                    _json_value(record),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(arrays_temp, arrays_path)
            os.replace(marker_temp, marker_path)
        finally:
            arrays_temp.unlink(missing_ok=True)
            marker_temp.unlink(missing_ok=True)


class BandExecutor:
    """Run one deterministic k-point shard with finite checkpointing."""

    def __init__(
        self,
        calculator: BandCalculator,
        journal_root: str | Path,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.calculator = calculator
        self.journal_root = Path(journal_root)
        self.rank = rank
        self.world_size = world_size

    def run(
        self,
        bundle: LocalizedOperatorBundle,
        request: BandRequest,
    ) -> tuple[BandPoint, ...]:
        indices = shard_indices(
            request.kpath.n_points,
            self.rank,
            self.world_size,
        )
        prepared = self.calculator.prepare(bundle, request)
        journal = BandPointJournal(
            self.journal_root,
            band_run_specification(
                bundle,
                request,
                self.calculator.solver,
                world_size=self.world_size,
            ),
        )
        completed: list[BandPoint] = []
        for index in indices:
            point = journal.load_point(
                index,
                request.kpath.fractional_points[index],
            )
            if point is None:
                print(
                    f"[rank {self.rank}] solving k-point "
                    f"{index + 1}/{request.kpath.n_points}",
                    flush=True,
                )
                point = prepared.solve_point(index)
                journal.write_point(point)
            else:
                print(
                    f"[rank {self.rank}] restored k-point "
                    f"{index + 1}/{request.kpath.n_points}",
                    flush=True,
                )
            completed.append(point)
        return tuple(completed)


def merge_band_journals(
    prepared: PreparedBandCalculation,
    journal_roots: Iterable[str | Path],
) -> BandResult:
    """Validate exact coverage and restore original path order."""

    roots = tuple(Path(root) for root in journal_roots)
    specification = band_run_specification(
        prepared.bundle,
        prepared.request,
        prepared.solver,
        world_size=len(roots),
    )
    owners: dict[int, BandPoint] = {}
    for root in roots:
        journal = BandPointJournal(root, specification)
        for index in range(prepared.request.kpath.n_points):
            point = journal.load_point(
                index,
                prepared.request.kpath.fractional_points[index],
            )
            if point is None:
                continue
            if index in owners:
                raise ArtifactError(f"Band point {index} has multiple owners")
            owners[index] = point
    expected = set(range(prepared.request.kpath.n_points))
    if set(owners) != expected:
        missing = sorted(expected - set(owners))
        raise ArtifactError(f"Band journals do not cover all points: {missing}")
    return prepared.build_result(owners.values())

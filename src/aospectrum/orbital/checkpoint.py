"""Explicit checkpoints for one interrupted Orbital calculation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from aospectrum.errors import ArtifactError
from aospectrum.model.spectra import (
    AbsoluteStateRange,
    NumericalQuality,
    QualityPolicy,
    SolverReceipt,
)
from aospectrum.orbital.model import (
    OrbitalEigensystem,
    OrbitalStateField,
    ResolvedStateSelection,
)
from aospectrum.orbital.volume import OrbitalField


RUN_SCHEMA = "aospectrum.orbital-journal/v1"
EIGENSYSTEM_SCHEMA = "aospectrum.orbital-eigensystem-checkpoint/v1"
FIELD_SCHEMA = "aospectrum.orbital-field-checkpoint/v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}-{uuid4().hex}")
    temporary.write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class OrbitalJournal:
    """One immutable request with committed eigensystem and field records."""

    def __init__(
        self,
        root: str | Path,
        specification: Mapping[str, Any],
        *,
        resume: bool,
    ) -> None:
        self.root = Path(root)
        normalized = dict(specification)
        normalized["schema"] = RUN_SCHEMA
        self.fields_root = self.root / "fields"
        if resume:
            try:
                observed = json.loads(
                    (self.root / "run.json").read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactError(
                    f"cannot read Orbital checkpoint {self.root}"
                ) from exc
            if _canonical_json(observed) != _canonical_json(normalized):
                raise ArtifactError(
                    "Orbital checkpoint belongs to another request"
                )
        else:
            if self.root.exists():
                raise ArtifactError(
                    f"Orbital checkpoint already exists: {self.root}"
                )
            self.fields_root.mkdir(parents=True)
            _write_json_atomic(self.root / "run.json", normalized)
        self.fields_root.mkdir(parents=True, exist_ok=True)

    def load_eigensystem(self) -> OrbitalEigensystem | None:
        arrays_path = self.root / "eigensystem.npz"
        marker_path = self.root / "eigensystem.json"
        if not marker_path.exists():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("schema") != EIGENSYSTEM_SCHEMA
                or not arrays_path.is_file()
                or marker.get("arrays_sha256") != _sha256(arrays_path)
            ):
                raise ArtifactError("Orbital eigensystem checkpoint is invalid")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                states = np.array(arrays["absolute_state_numbers"], copy=True)
                energies = np.array(arrays["energies_ev"], copy=True)
                vectors = np.array(arrays["eigenvectors"], copy=True)
                anchors = np.array(arrays["phase_anchor_indices"], copy=True)
                raw = np.array(arrays["raw_residuals_ev"], copy=True)
                normalized = np.array(
                    arrays["normalized_residuals"],
                    copy=True,
                )
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                "cannot decode Orbital eigensystem checkpoint"
            ) from exc
        try:
            selection = marker["selection"]
            quality = marker["quality"]
            receipt = marker["receipt"]
            return OrbitalEigensystem(
                selection=ResolvedStateSelection(
                    expression=selection["expression"],
                    absolute=AbsoluteStateRange(
                        selection["first"],
                        selection["last"],
                    ),
                    semantics=selection["semantics"],
                ),
                absolute_state_numbers=states,
                energies_ev=energies,
                eigenvectors=vectors,
                phase_anchor_indices=anchors,
                quality=NumericalQuality(
                    raw_residuals_ev=raw,
                    normalized_residuals=normalized,
                    s_orthogonality_error=quality["s_orthogonality_error"],
                    calculation_dtype=quality["calculation_dtype"],
                    policy=QualityPolicy(**quality["policy"]),
                ),
                receipt=SolverReceipt(
                    backend=receipt["backend"],
                    device=receipt["device"],
                    scalar_dtype=receipt["scalar_dtype"],
                    stage_seconds=receipt["stage_seconds"],
                    peak_host_rss_bytes=receipt["peak_host_rss_bytes"],
                    peak_gpu_memory_bytes=receipt["peak_gpu_memory_bytes"],
                    counters=receipt["counters"],
                    details=receipt["details"],
                ),
                backend=marker["backend"],
                degeneracy_notices=tuple(marker["degeneracy_notices"]),
                locator_evidence=marker["locator_evidence"],
                metadata=marker["metadata"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(
                "Orbital eigensystem checkpoint metadata is invalid"
            ) from exc

    def write_eigensystem(self, value: OrbitalEigensystem) -> None:
        if self.load_eigensystem() is not None:
            return
        arrays_path = self.root / "eigensystem.npz"
        arrays_temp = self.root / f".eigensystem-{uuid4().hex}.npz"
        np.savez(
            arrays_temp,
            absolute_state_numbers=value.absolute_state_numbers,
            energies_ev=value.energies_ev,
            eigenvectors=value.eigenvectors,
            phase_anchor_indices=value.phase_anchor_indices,
            raw_residuals_ev=value.quality.raw_residuals_ev,
            normalized_residuals=value.quality.normalized_residuals,
        )
        os.replace(arrays_temp, arrays_path)
        selection = value.selection
        quality = value.quality
        receipt = value.receipt
        _write_json_atomic(
            self.root / "eigensystem.json",
            {
                "schema": EIGENSYSTEM_SCHEMA,
                "arrays_sha256": _sha256(arrays_path),
                "selection": {
                    "expression": selection.expression,
                    "semantics": selection.semantics,
                    "first": selection.absolute.first,
                    "last": selection.absolute.last,
                },
                "quality": {
                    "s_orthogonality_error": quality.s_orthogonality_error,
                    "calculation_dtype": quality.calculation_dtype,
                    "policy": {
                        "name": quality.policy.name,
                        "raw_residual_tolerance_ev": (
                            quality.policy.raw_residual_tolerance_ev
                        ),
                        "normalized_residual_tolerance": (
                            quality.policy.normalized_residual_tolerance
                        ),
                        "s_orthogonality_tolerance": (
                            quality.policy.s_orthogonality_tolerance
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
                "backend": value.backend,
                "degeneracy_notices": list(value.degeneracy_notices),
                "locator_evidence": dict(value.locator_evidence),
                "metadata": dict(value.metadata),
            },
        )

    def _field_paths(self, state_number: int) -> tuple[Path, Path]:
        stem = f"state-{int(state_number):06d}"
        return self.fields_root / f"{stem}.npz", self.fields_root / f"{stem}.json"

    def load_field(
        self,
        state_number: int,
        *,
        energy_ev: float,
    ) -> OrbitalStateField | None:
        arrays_path, marker_path = self._field_paths(state_number)
        if not marker_path.exists():
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                marker.get("schema") != FIELD_SCHEMA
                or marker.get("state_number") != int(state_number)
                or not isinstance(marker.get("label"), str)
                or not marker["label"]
                or float(marker.get("energy_ev")) != float(energy_ev)
                or not arrays_path.is_file()
                or marker.get("arrays_sha256") != _sha256(arrays_path)
            ):
                raise ArtifactError("Orbital field checkpoint is invalid")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                values = np.array(arrays["values"], copy=True)
                cell = np.array(arrays["cell_angstrom"], copy=True)
                origin = np.array(arrays["origin_angstrom"], copy=True)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"cannot decode Orbital field checkpoint for state {state_number}"
            ) from exc
        try:
            return OrbitalStateField(
                state_number=state_number,
                label=marker["label"],
                energy_ev=energy_ev,
                field=OrbitalField(
                    values=values,
                    grid_shape=tuple(marker["grid_shape"]),
                    cell_angstrom=cell,
                    origin_angstrom=origin,
                    voxel_volume_bohr3=marker["voxel_volume_bohr3"],
                    compute_device=marker["compute_device"],
                    precision=marker["precision"],
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(
                f"Orbital field checkpoint metadata is invalid for "
                f"state {state_number}"
            ) from exc

    def write_field(self, value: OrbitalStateField) -> None:
        arrays_path, marker_path = self._field_paths(value.state_number)
        if self.load_field(
            value.state_number,
            energy_ev=value.energy_ev,
        ) is not None:
            return
        arrays_temp = self.fields_root / f".{arrays_path.name}-{uuid4().hex}"
        marker_temp = self.fields_root / f".{marker_path.name}-{uuid4().hex}"
        try:
            with arrays_temp.open("wb") as handle:
                np.savez(
                    handle,
                    values=value.field.values,
                    cell_angstrom=value.field.cell_angstrom,
                    origin_angstrom=value.field.origin_angstrom,
                )
            marker = {
                "schema": FIELD_SCHEMA,
                "state_number": value.state_number,
                "label": value.label,
                "energy_ev": value.energy_ev,
                "arrays_sha256": _sha256(arrays_temp),
                "grid_shape": list(value.field.grid_shape),
                "voxel_volume_bohr3": value.field.voxel_volume_bohr3,
                "compute_device": value.field.compute_device,
                "precision": value.field.precision,
            }
            marker_temp.write_text(
                json.dumps(marker, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(arrays_temp, arrays_path)
            os.replace(marker_temp, marker_path)
        finally:
            arrays_temp.unlink(missing_ok=True)
            marker_temp.unlink(missing_ok=True)

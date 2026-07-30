"""Configured selected-state Orbital application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from aospectrum import __version__, load_bundle
from aospectrum.cli.config import (
    bind_cuda_visible_device,
    load_config,
    reject_unknown_keys,
    required_string,
)
from aospectrum.errors import ArtifactError, ContractError
from aospectrum.orbital import (
    GridSpec,
    OrbitalCalculator,
    OrbitalRequest,
    write_orbital_viewer,
)
from aospectrum.orbital.checkpoint import OrbitalJournal
from aospectrum.orbital.providers import FieldResources, OpenMXPAOProvider
from aospectrum.orbital.providers.openmx_pao_data import read_openmx_pao
from aospectrum.solvers import create_orbital_solver


@dataclass(frozen=True, slots=True)
class OrbitalRunConfiguration:
    source: Path
    bundle: Path
    request: OrbitalRequest
    device: str
    basis_definition: Path
    chunk_points: int
    default_representation: str
    enclosed_probability: float
    output: Path

    @property
    def checkpoint_root(self) -> Path:
        return self.output.with_name(f".{self.output.name}.checkpoints")


def read_orbital_configuration(path: str | Path) -> OrbitalRunConfiguration:
    document = load_config(path)
    document.require_sections(
        {
            "input",
            "orbital",
            "basis",
            "compute",
            "resources",
            "viewer",
            "output",
        }
    )
    input_section = document.section("input")
    orbital_section = document.section("orbital")
    basis_section = document.section("basis")
    compute_section = document.section("compute")
    resources_section = document.section("resources")
    viewer_section = document.section("viewer")
    output_section = document.section("output")
    reject_unknown_keys(input_section, {"bundle"}, context="input")
    reject_unknown_keys(
        orbital_section,
        {
            "states",
            "kpoint",
            "grid_shape",
            "grid_spacing_angstrom",
            "energy_hint_ev",
        },
        context="orbital",
    )
    reject_unknown_keys(basis_section, {"definition"}, context="basis")
    reject_unknown_keys(compute_section, {"device"}, context="compute")
    reject_unknown_keys(
        resources_section,
        {"chunk_points"},
        context="resources",
    )
    reject_unknown_keys(
        viewer_section,
        {"default_representation", "enclosed_probability"},
        context="viewer",
    )
    reject_unknown_keys(output_section, {"directory"}, context="output")
    kpoint = orbital_section.get("kpoint", [0.0, 0.0, 0.0])
    try:
        kpoint_array = np.asarray(kpoint, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContractError("orbital.kpoint must contain three numbers") from exc
    if kpoint_array.shape != (3,) or not np.array_equal(
        kpoint_array,
        np.zeros(3),
    ):
        raise ContractError("v1 Orbital requires Gamma kpoint = [0, 0, 0]")
    grid = _read_grid(orbital_section)
    device = _compute_device(compute_section.get("device"))
    chunk_points = resources_section.get("chunk_points", 65_536)
    if (
        isinstance(chunk_points, bool)
        or not isinstance(chunk_points, int)
        or chunk_points <= 0
    ):
        raise ContractError("resources.chunk_points must be positive")
    default_representation = required_string(
        viewer_section,
        "default_representation",
        context="viewer",
    )
    if default_representation not in {"phase", "density"}:
        raise ContractError(
            "viewer.default_representation must be phase or density"
        )
    probability = viewer_section.get("enclosed_probability", 0.80)
    if isinstance(probability, bool):
        raise ContractError("viewer.enclosed_probability must be numeric")
    try:
        probability = float(probability)
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "viewer.enclosed_probability must be numeric"
        ) from exc
    if not math.isfinite(probability) or not 0.50 <= probability <= 0.99:
        raise ContractError(
            "viewer.enclosed_probability must be between 0.50 and 0.99"
        )
    energy_hint = orbital_section.get("energy_hint_ev")
    try:
        resolved_energy_hint = (
            None if energy_hint is None else float(energy_hint)
        )
    except (TypeError, ValueError) as exc:
        raise ContractError("orbital.energy_hint_ev must be numeric") from exc
    return OrbitalRunConfiguration(
        source=document.path,
        bundle=document.resolve_path(
            required_string(input_section, "bundle", context="input"),
            name="input.bundle",
        ),
        request=OrbitalRequest(
            states=required_string(
                orbital_section,
                "states",
                context="orbital",
            ),
            precision="float32",
            grid=grid,
            energy_hint_ev=resolved_energy_hint,
        ),
        device=device,
        basis_definition=document.resolve_path(
            required_string(
                basis_section,
                "definition",
                context="basis",
            ),
            name="basis.definition",
        ),
        chunk_points=chunk_points,
        default_representation=default_representation,
        enclosed_probability=probability,
        output=document.resolve_path(
            required_string(output_section, "directory", context="output"),
            name="output.directory",
        ),
    )


def _compute_device(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("compute.device must be a nonnegative GPU index")
    return f"cuda:{value}"


def _read_grid(section: Mapping[str, Any]) -> GridSpec:
    shape = section.get("grid_shape")
    spacing = section.get("grid_spacing_angstrom")
    if (shape is None) == (spacing is None):
        raise ContractError(
            "orbital requires exactly one of grid_shape or "
            "grid_spacing_angstrom"
        )
    if shape is not None:
        if not isinstance(shape, list) or len(shape) != 3:
            raise ContractError("orbital.grid_shape must contain three integers")
        return GridSpec(shape=tuple(shape))
    if isinstance(spacing, bool):
        raise ContractError("orbital.grid_spacing_angstrom must be numeric")
    try:
        return GridSpec(spacing_angstrom=float(spacing))
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "orbital.grid_spacing_angstrom must be numeric"
        ) from exc


def _read_openmx_provider(
    definition_path: Path,
    expected_basis_identity: str,
) -> tuple[OpenMXPAOProvider, dict[str, str]]:
    definition = load_config(definition_path)
    identity = definition.values.get("basis_identity")
    if identity != expected_basis_identity:
        raise ContractError(
            "OpenMX definition basis_identity differs from operator bundle"
        )
    files = definition.values.get("pao_files")
    if not isinstance(files, dict) or not files:
        raise ContractError("OpenMX definition requires [pao_files]")
    pao_by_atomic_number = {}
    pao_sha256: dict[str, str] = {}
    for atomic_number, value in files.items():
        try:
            number = int(atomic_number)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "OpenMX PAO keys must be atomic numbers"
            ) from exc
        path = definition.resolve_path(
            value,
            name=f"pao_files.{atomic_number}",
        )
        pao_by_atomic_number[number] = read_openmx_pao(path)
        pao_sha256[str(number)] = _sha256(path)
    return (
        OpenMXPAOProvider(
            basis_identity=expected_basis_identity,
            pao_by_atomic_number=pao_by_atomic_number,
        ),
        pao_sha256,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _orbital_run_specification(
    configuration: OrbitalRunConfiguration,
    bundle_identity: str,
    basis_identity: str,
    pao_sha256: Mapping[str, str],
) -> dict[str, Any]:
    grid = configuration.request.grid
    return {
        "aospectrum_version": __version__,
        "bundle_identity": bundle_identity,
        "basis_identity": basis_identity,
        "config_sha256": _sha256(configuration.source),
        "basis_definition_sha256": _sha256(
            configuration.basis_definition
        ),
        "pao_sha256": dict(sorted(pao_sha256.items())),
        "states": configuration.request.states,
        "grid": {
            "shape": None if grid.shape is None else list(grid.shape),
            "spacing_angstrom": grid.spacing_angstrom,
        },
        "energy_hint_ev": configuration.request.energy_hint_ev,
        "device": configuration.device,
        "chunk_points": configuration.chunk_points,
    }


def _write_run_status(
    root: Path,
    *,
    status: str,
    error: str | None = None,
) -> None:
    temporary = root / f".status-{uuid4().hex}.json"
    temporary.write_text(
        json.dumps(
            {
                "schema": "aospectrum.orbital-run-status/v1",
                "status": status,
                "aospectrum_version": __version__,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, root / "status.json")


def _require_run_paths(
    configuration: OrbitalRunConfiguration,
    *,
    resume: bool,
) -> None:
    output = configuration.output
    if output.exists():
        if not output.is_dir():
            raise ArtifactError(
                f"Orbital output path is not a directory: {output}"
            )
        if any(output.iterdir()):
            raise ArtifactError(f"Orbital output is not empty: {output}")
    checkpoint = configuration.checkpoint_root
    if resume:
        if not checkpoint.is_dir():
            raise ArtifactError(
                f"Orbital resume checkpoint does not exist: {checkpoint}"
            )
    elif checkpoint.exists():
        raise ArtifactError(
            f"Orbital checkpoint already exists: {checkpoint}; "
            "use --resume or choose a new output directory"
        )


def run_orbital_config(path: str | Path, *, resume: bool = False) -> Path:
    configuration = read_orbital_configuration(path)
    _require_run_paths(configuration, resume=resume)
    if configuration.device.startswith("cuda:"):
        bind_cuda_visible_device(configuration.device)
    bundle = load_bundle(configuration.bundle)
    provider, pao_sha256 = _read_openmx_provider(
        configuration.basis_definition,
        bundle.basis.basis_identity,
    )
    bundle_identity = str(
        bundle.provenance.get(
            "bundle_manifest_sha256",
            (
                f"in-memory:{bundle.basis.basis_identity}:"
                f"{bundle.n_orbitals}:{bundle.operators.n_blocks}"
            ),
        )
    )
    journal = OrbitalJournal(
        configuration.checkpoint_root,
        _orbital_run_specification(
            configuration,
            bundle_identity,
            bundle.basis.basis_identity,
            pao_sha256,
        ),
        resume=resume,
    )
    _write_run_status(configuration.checkpoint_root, status="running")
    calculator: OrbitalCalculator | None = None
    try:
        solver = create_orbital_solver()
        calculator = OrbitalCalculator(solver)
        eigensystem = journal.load_eigensystem()
        if eigensystem is None:
            print("[orbital] solving selected eigensystem", flush=True)
            eigensystem = calculator.solve_eigensystem(
                bundle,
                configuration.request,
            )
            journal.write_eigensystem(eigensystem)
        else:
            print("[orbital] restored selected eigensystem", flush=True)
        resources = FieldResources(
            device="cuda:0",
            chunk_points=configuration.chunk_points,
        )

        def fields():
            for column, state_value in enumerate(
                eigensystem.absolute_state_numbers
            ):
                state = int(state_value)
                energy = float(eigensystem.energies_ev[column])
                restored = journal.load_field(state, energy_ev=energy)
                if restored is None:
                    print(
                        f"[orbital] evaluating {column + 1}/"
                        f"{eigensystem.absolute_state_numbers.size}: "
                        f"state {state}",
                        flush=True,
                    )
                    restored = calculator.evaluate_state(
                        bundle,
                        configuration.request,
                        eigensystem,
                        provider,
                        resources,
                        column=column,
                    )
                    journal.write_field(restored)
                else:
                    print(
                        f"[orbital] restored state {state}",
                        flush=True,
                    )
                yield restored

        output = write_orbital_viewer(
            configuration.output,
            bundle.structure,
            eigensystem,
            fields(),
            default_probability=configuration.enclosed_probability,
            default_representation=configuration.default_representation,
        )
    except Exception as exc:
        _write_run_status(
            configuration.checkpoint_root,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        if calculator is not None:
            calculator.close()
    shutil.rmtree(configuration.checkpoint_root)
    print(output)
    return output

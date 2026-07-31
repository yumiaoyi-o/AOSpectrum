"""Integrated multi-state NGL viewer artifact writer."""

from __future__ import annotations

from importlib import resources
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
from uuid import uuid4

import numpy as np

from .data import AtomicStructure
from .errors import AOSpectrumError, InputError, SolverError
from .orbital import (
    OrbitalEigensystem,
    OrbitalStateField,
    enclosed_probability_levels,
)


VIEWER_SCHEMA = "aospectrum.orbital-viewer/v1"


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
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


def _matrix_rows(field) -> list[list[float]]:
    nx, ny, nz = field.grid_shape
    a, b, c = field.cell_angstrom
    origin = field.origin_angstrom
    return [
        [a[0] / nx, b[0] / ny, c[0] / nz, origin[0]],
        [a[1] / nx, b[1] / ny, c[1] / nz, origin[1]],
        [a[2] / nx, b[2] / ny, c[2] / nz, origin[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_f32(path: Path, values: np.ndarray) -> None:
    shaped = np.asarray(values, dtype=np.dtype("<f4"))
    np.asfortranarray(shaped).ravel(order="F").tofile(path)


def write_orbital_viewer(
    destination: str | Path,
    structure: AtomicStructure,
    eigensystem: OrbitalEigensystem,
    state_fields: Iterable[OrbitalStateField],
    *,
    default_probability: float = 0.80,
    default_representation: str = "phase",
) -> Path:
    """Write one integrated viewer while consuming fields one at a time."""

    if (
        not math.isfinite(default_probability)
        or not 0.50 <= default_probability <= 0.99
    ):
        raise InputError(
            "default enclosed probability must be between 0.50 and 0.99"
        )
    if default_representation not in {"phase", "density"}:
        raise InputError(
            "default representation must be 'phase' or 'density'"
        )
    root = Path(destination)
    if root.exists():
        if not root.is_dir():
            raise InputError(
                f"Orbital destination is not a directory: {root}"
            )
        if any(root.iterdir()):
            raise InputError(
                f"Orbital destination is not empty: {root}"
            )
    temporary_root = root.with_name(f".{root.name}.tmp-{uuid4().hex}")
    states_root = temporary_root / "states"
    assets_root = temporary_root / "assets"
    states_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)
    expected_states = eigensystem.state_numbers.tolist()
    records: list[dict[str, Any]] = []
    observed: list[int] = []
    try:
        for state_field in state_fields:
            state = state_field.state_number
            observed.append(state)
            state_id = f"state-{state:06d}"
            state_root = states_root / state_id
            state_root.mkdir()
            phase = np.real(state_field.field.values)
            density = np.abs(state_field.field.values) ** 2
            _write_f32(state_root / "phase.f32", phase)
            _write_f32(state_root / "density.f32", density)
            levels = enclosed_probability_levels(state_field.field)
            metadata = {
                "schema": "aospectrum.orbital-volume/v1",
                "state_number": state,
                "label": state_field.label,
                "energy_ev": state_field.energy_ev,
                "grid": list(state_field.field.grid_shape),
                "matrix_rows": _matrix_rows(state_field.field),
                "voxel_volume_bohr3": (
                    state_field.field.voxel_volume_bohr3
                ),
                "captured_norm": levels.captured_norm,
                "compute_device": state_field.field.compute_device,
                "field_precision": "complex64",
                "isovalue_mode": "enclosed_probability",
                "probabilities": levels.probabilities.tolist(),
                "amplitude_levels": levels.amplitude_levels.tolist(),
                "density_levels": levels.density_levels.tolist(),
                "phase_path": "phase.f32",
                "density_path": "density.f32",
            }
            _write_json(state_root / "metadata.json", metadata)
            records.append(
                {
                    "id": state_id,
                    "state_number": state,
                    "label": state_field.label,
                    "energy_ev": state_field.energy_ev,
                    "evaluation_seconds": state_field.evaluation_seconds,
                    "metadata": f"states/{state_id}/metadata.json",
                    "phase": f"states/{state_id}/phase.f32",
                    "density": f"states/{state_id}/density.f32",
                }
            )
        if observed != expected_states:
            raise SolverError(
                "Orbital fields do not exactly cover selected states in order"
            )
        _write_json(
            temporary_root / "viewer-manifest.json",
            {
                "schema": VIEWER_SCHEMA,
                "default_probability": float(default_probability),
                "default_representation": default_representation,
                "calculation": {
                    "backend": "primme-cudss",
                    "scalar_dtype": "float32",
                    "selection": {
                        "expression": eigensystem.selection.expression,
                        "semantics": eigensystem.selection.semantics,
                        "first": eigensystem.selection.states.first,
                        "last": eigensystem.selection.states.last,
                    },
                    "maximum_eigenpair_residual_ev": (
                        eigensystem.maximum_residual_ev
                    ),
                    "stage_seconds": eigensystem.stage_seconds,
                    "peak_gpu_memory_bytes": (
                        eigensystem.peak_gpu_memory_bytes
                    ),
                    "field_evaluation_seconds": sum(
                        record["evaluation_seconds"]
                        for record in records
                    ),
                    "warnings": list(eigensystem.warnings),
                },
                "structure": {
                    "positions_angstrom": (
                        structure.positions_angstrom.tolist()
                    ),
                    "atomic_numbers": structure.atomic_numbers.tolist(),
                    "cell_angstrom": structure.cell_angstrom.tolist(),
                    "origin_angstrom": [0.0, 0.0, 0.0],
                    "periodic": list(structure.periodic),
                },
                "states": records,
            },
        )
        (temporary_root / "index.html").write_text(
            _viewer_html(),
            encoding="utf-8",
        )
        asset_package = resources.files("aospectrum.assets")
        with resources.as_file(asset_package / "ngl-2.4.0.js") as source:
            shutil.copyfile(source, assets_root / "ngl-2.4.0.js")
        with resources.as_file(asset_package / "NGL-LICENSE.txt") as source:
            shutil.copyfile(source, assets_root / "NGL-LICENSE.txt")
    except Exception as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        if isinstance(exc, AOSpectrumError):
            raise
        raise SolverError(f"cannot write Orbital viewer {root}: {exc}") from exc
    try:
        if root.exists():
            root.rmdir()
        os.replace(temporary_root, root)
    except OSError as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise SolverError(f"cannot publish Orbital viewer {root}: {exc}") from exc
    return root


def _viewer_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AOSpectrum Orbital Viewer</title>
  <script src="assets/ngl-2.4.0.js"></script>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; color: #171a20; background: #f4f5f7; }
    header { min-height: 58px; display: flex; align-items: center; gap: 16px;
      padding: 10px 16px; border-bottom: 1px solid #cfd3da; background: #fff; }
    h1 { margin: 0 18px 0 0; font-size: 18px; font-weight: 650; }
    label { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; }
    select, input, button { font: inherit; }
    select, button { min-height: 34px; border: 1px solid #aeb5bf; border-radius: 4px;
      background: #fff; color: #171a20; padding: 5px 9px; }
    button { cursor: pointer; }
    input[type=range] { width: 150px; }
    #level-label { min-width: 118px; font-variant-numeric: tabular-nums; }
    #status { margin-left: auto; color: #5a6370; font-size: 13px; }
    #subtitle { height: 34px; padding: 8px 16px; border-bottom: 1px solid #d8dce2;
      background: #fff; color: #4d5663; font-size: 13px; }
    #viewport { width: 100vw; height: calc(100vh - 92px); background: #fff; }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-wrap: wrap; }
      #status { width: 100%; margin-left: 0; }
      #viewport { height: calc(100vh - 150px); }
    }
  </style>
</head>
<body>
  <header>
    <h1>AOSpectrum Orbital Viewer</h1>
    <label>State <select id="state"></select></label>
    <label>Representation
      <select id="representation">
        <option value="phase">Phase-fixed Re(psi)</option>
        <option value="density">Density |psi|^2</option>
      </select>
    </label>
    <label>Enclosed probability
      <input id="probability" type="range" min="0.50" max="0.99" step="0.01" value="0.80">
      <span id="level-label"></span>
    </label>
    <label><input id="atoms" type="checkbox" checked>Atoms</label>
    <label><input id="cell" type="checkbox" checked>Cell</label>
    <button id="reset" type="button">Reset view</button>
    <button id="snapshot" type="button">Snapshot</button>
    <span id="status">Loading</span>
  </header>
  <div id="subtitle"></div>
  <div id="viewport"></div>
  <script>
  "use strict";
  const stage = new NGL.Stage("viewport", { backgroundColor: "white" });
  const stateSelect = document.getElementById("state");
  const representation = document.getElementById("representation");
  const probability = document.getElementById("probability");
  const levelLabel = document.getElementById("level-label");
  const status = document.getElementById("status");
  const subtitle = document.getElementById("subtitle");
  let manifest, metadata, positiveSurface, negativeSurface, densitySurface;
  let atomRepresentation, cellRepresentation;

  const matrixFromRows = rows => new NGL.Matrix4().set(
    rows[0][0], rows[0][1], rows[0][2], rows[0][3],
    rows[1][0], rows[1][1], rows[1][2], rows[1][3],
    rows[2][0], rows[2][1], rows[2][2], rows[2][3],
    rows[3][0], rows[3][1], rows[3][2], rows[3][3]
  );
  const add = (...vectors) => vectors.reduce(
    (sum, vector) => sum.map((value, axis) => value + vector[axis]),
    [0, 0, 0]
  );
  const atomStyle = z => ({
    1: [[1, 1, 1], 1.20], 6: [[0.36, 0.36, 0.36], 1.70],
    7: [[0.19, 0.31, 0.97], 1.55], 8: [[1, 0.05, 0.05], 1.52],
    14: [[0.94, 0.78, 0.63], 2.10], 15: [[1, 0.5, 0], 1.80],
    16: [[1, 0.9, 0.19], 1.80]
  }[z] || [[0.20, 0.63, 0.62], 1.80]);
  const fetchFloat32 = async path => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Cannot load ${path}`);
    return new Float32Array(await response.arrayBuffer());
  };
  const makeVolume = (name, data) => {
    const expected = metadata.grid.reduce((a, b) => a * b, 1);
    if (data.length !== expected) throw new Error(`${name} volume size mismatch`);
    const volume = new NGL.Volume(
      name, "", data, metadata.grid[0], metadata.grid[1], metadata.grid[2]
    );
    volume.setMatrix(matrixFromRows(metadata.matrix_rows));
    return stage.addComponentFromObject(volume, { name });
  };
  const levelIndex = () => {
    const target = Number(probability.value);
    let best = 0;
    metadata.probabilities.forEach((value, index) => {
      if (Math.abs(value - target) < Math.abs(metadata.probabilities[best] - target)) best = index;
    });
    return best;
  };
  const surfaceParameters = (isolevel, color, negateIsolevel = false) => ({
    isolevelType: "value", isolevel, negateIsolevel, color, opacity: 0.82,
    side: "double", smooth: 1, useWorker: true, wrap: true, opaqueBack: false,
    quality: "high"
  });
  const addStructure = () => {
    const payload = manifest.structure;
    const positions = new Float32Array(payload.positions_angstrom.flat());
    const colors = new Float32Array(positions.length);
    const radii = new Float32Array(payload.atomic_numbers.length);
    payload.atomic_numbers.forEach((z, index) => {
      const [color, radius] = atomStyle(z);
      colors.set(color, index * 3); radii[index] = radius * 0.18;
    });
    const atoms = new NGL.Shape("Atoms");
    atoms.addBuffer(new NGL.SphereBuffer({ position: positions, color: colors, radius: radii }));
    atomRepresentation = stage.addComponentFromObject(atoms).addRepresentation("buffer");
    const [a, b, c] = payload.cell_angstrom, origin = payload.origin_angstrom;
    const corners = [origin, add(origin,a), add(origin,b), add(origin,a,b),
      add(origin,c), add(origin,a,c), add(origin,b,c), add(origin,a,b,c)];
    const edges = [[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],
      [4,5],[4,6],[5,7],[6,7]];
    const cell = new NGL.Shape("Cell");
    edges.forEach(([start,end]) => cell.addCylinder(
      corners[start], corners[end], [0.15,0.17,0.20], 0.06
    ));
    cellRepresentation = stage.addComponentFromObject(cell).addRepresentation("buffer");
  };
  const updateLevel = () => {
    const index = levelIndex(), showPhase = representation.value === "phase";
    const level = showPhase ? metadata.amplitude_levels[index] : metadata.density_levels[index];
    levelLabel.textContent = `${Math.round(metadata.probabilities[index] * 100)}% captured | ${level.toExponential(2)}`;
    if (positiveSurface) positiveSurface.setParameters({ isolevel: metadata.amplitude_levels[index] });
    if (negativeSurface) negativeSurface.setParameters({ isolevel: metadata.amplitude_levels[index] });
    if (densitySurface) densitySurface.setParameters({ isolevel: metadata.density_levels[index] });
  };
  const applyRepresentation = () => {
    const phase = representation.value === "phase";
    positiveSurface.setVisibility(phase); negativeSurface.setVisibility(phase);
    densitySurface.setVisibility(!phase); updateLevel();
  };
  const loadState = async () => {
    status.textContent = "Loading state";
    stage.removeAllComponents();
    const state = manifest.states[stateSelect.selectedIndex];
    metadata = await (await fetch(state.metadata)).json();
    const [phase, density] = await Promise.all([
      fetchFloat32(state.phase), fetchFloat32(state.density)
    ]);
    const index = levelIndex();
    const phaseComponent = makeVolume("Phase", phase);
    positiveSurface = phaseComponent.addRepresentation(
      "surface", surfaceParameters(metadata.amplitude_levels[index], "#d43f3a")
    );
    negativeSurface = phaseComponent.addRepresentation(
      "surface", surfaceParameters(metadata.amplitude_levels[index], "#2563c7", true)
    );
    densitySurface = makeVolume("Density", density).addRepresentation(
      "surface", surfaceParameters(metadata.density_levels[index], "#168a61")
    );
    addStructure();
    atomRepresentation.setVisibility(document.getElementById("atoms").checked);
    cellRepresentation.setVisibility(document.getElementById("cell").checked);
    applyRepresentation();
    subtitle.textContent = `${state.label} | state ${state.state_number} | ` +
      `${Number(state.energy_ev).toFixed(6)} eV | ${metadata.grid.join(" x ")} | ` +
      `captured grid norm ${Number(metadata.captured_norm).toFixed(5)}`;
    stage.autoView(0); status.textContent = "Ready";
  };
  const initialise = async () => {
    manifest = await (await fetch("viewer-manifest.json")).json();
    manifest.states.forEach(state => stateSelect.add(new Option(state.label, state.id)));
    probability.value = String(manifest.default_probability);
    representation.value = manifest.default_representation;
    await loadState();
  };
  stateSelect.addEventListener("change", loadState);
  representation.addEventListener("change", applyRepresentation);
  probability.addEventListener("input", updateLevel);
  document.getElementById("atoms").addEventListener("change", event =>
    atomRepresentation.setVisibility(event.target.checked));
  document.getElementById("cell").addEventListener("change", event =>
    cellRepresentation.setVisibility(event.target.checked));
  document.getElementById("reset").addEventListener("click", () => stage.autoView(250));
  document.getElementById("snapshot").addEventListener("click", async () => {
    const blob = await stage.makeImage({ factor: 2, antialias: true, trim: false });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob); link.download = "aospectrum-orbital.png";
    link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  });
  window.addEventListener("resize", () => stage.handleResize());
  initialise().catch(error => { status.textContent = error.message; console.error(error); });
  </script>
</body>
</html>
"""

"""AOSpectrum command-line applications."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from . import __version__
from .band import (
    BandCalculation,
    KPath,
    read_kpath,
    read_point,
    resolve_energy_reference,
    write_band_output,
    write_point,
)
from .bundle import load_bundle
from .ciss import CissSettings
from .data import EnergyInterval, LocalizedOperatorBundle
from .errors import AOSpectrumError, InputError, SolverError
from .ngl import write_orbital_viewer
from .openmx import OpenMXPAOProvider
from .openmx_data import read_openmx_pao
from .orbital import (
    FieldResources,
    GridSpec,
    OrbitalCalculation,
    read_eigensystem,
    read_state_field,
    write_eigensystem,
    write_state_field,
)


@dataclass(frozen=True, slots=True)
class BandConfig:
    source: Path
    bundle: Path
    kpath: KPath
    interval: EnergyInterval
    reference_mode: str
    reference_value_ev: float | None
    devices: tuple[int, ...]
    settings: CissSettings
    output: Path

    @property
    def work(self) -> Path:
        return self.output.with_name(f".{self.output.name}.work")


@dataclass(frozen=True, slots=True)
class OrbitalConfig:
    source: Path
    bundle: Path
    states: str
    grid: GridSpec
    energy_hint_ev: float | None
    basis_definition: Path
    device: int
    chunk_points: int
    default_representation: str
    enclosed_probability: float
    output: Path

    @property
    def work(self) -> Path:
        return self.output.with_name(f".{self.output.name}.work")


def _toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("rb") as handle:
            values = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InputError(f"cannot read TOML configuration {source}: {exc}") from exc
    return source, values


def _section(values: dict[str, Any], name: str) -> dict[str, Any]:
    section = values.get(name, {})
    if not isinstance(section, dict):
        raise InputError(f"[{name}] must be a TOML table")
    return section


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise InputError(f"{name} must contain two numbers")
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must contain two numbers") from exc


def read_band_config(path: str | Path) -> BandConfig:
    source, values = _toml(path)
    base = source.parent
    input_section = _section(values, "input")
    band = _section(values, "band")
    compute = _section(values, "compute")
    solver = _section(values, "solver")
    resources = _section(values, "resources")
    output = _section(values, "output")
    devices = compute.get("devices", [0])
    if (
        not isinstance(devices, list)
        or not devices
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in devices
        )
        or len(set(devices)) != len(devices)
    ):
        raise InputError("compute.devices must contain unique GPU indices")
    interval = EnergyInterval(
        *_pair(band.get("energy_interval_ev"), "band.energy_interval_ev")
    )
    reference_mode = str(band.get("energy_reference", "input_zero"))
    reference_value = band.get("energy_reference_ev")
    try:
        reference_value = (
            None if reference_value is None else float(reference_value)
        )
        settings = CissSettings(
            integration_points=int(solver.get("integration_points", 32)),
            contour_batch_size=int(
                solver.get(
                    "contour_batch_size",
                    resources.get("contour_batch_size", 1),
                )
            ),
            block_size=int(solver.get("block_size", 32)),
            moment_size=int(solver.get("moment_size", 4)),
        )
    except (TypeError, ValueError) as exc:
        raise InputError("invalid Band solver setting") from exc
    return BandConfig(
        source,
        _path(base, input_section.get("bundle"), "input.bundle"),
        read_kpath(_path(base, band.get("kpath"), "band.kpath")),
        interval,
        reference_mode,
        reference_value,
        tuple(devices),
        settings,
        _path(base, output.get("directory"), "output.directory"),
    )


def read_orbital_config(path: str | Path) -> OrbitalConfig:
    source, values = _toml(path)
    base = source.parent
    input_section = _section(values, "input")
    orbital = _section(values, "orbital")
    basis = _section(values, "basis")
    compute = _section(values, "compute")
    resources = _section(values, "resources")
    viewer = _section(values, "viewer")
    output = _section(values, "output")
    shape = orbital.get("grid_shape")
    spacing = orbital.get("grid_spacing_angstrom")
    if (shape is None) == (spacing is None):
        raise InputError(
            "orbital needs exactly one of grid_shape or "
            "grid_spacing_angstrom"
        )
    if shape is not None:
        if not isinstance(shape, list):
            raise InputError("orbital.grid_shape must be a list")
        grid = GridSpec(shape=tuple(shape))
    else:
        try:
            grid = GridSpec(spacing_angstrom=float(spacing))
        except (TypeError, ValueError) as exc:
            raise InputError(
                "orbital needs grid_shape or grid_spacing_angstrom"
            ) from exc
    try:
        kpoint = np.asarray(
            orbital.get("kpoint", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise InputError("orbital.kpoint must contain three numbers") from exc
    if kpoint.shape != (3,) or not np.array_equal(kpoint, np.zeros(3)):
        raise InputError("Orbital currently requires kpoint = [0, 0, 0]")
    device = compute.get("device", 0)
    chunk_points = resources.get("chunk_points", 65_536)
    if (
        isinstance(device, bool)
        or not isinstance(device, int)
        or device < 0
        or isinstance(chunk_points, bool)
        or not isinstance(chunk_points, int)
        or chunk_points < 1
    ):
        raise InputError("invalid Orbital compute resources")
    states = orbital.get("states")
    if not isinstance(states, str) or not states.strip():
        raise InputError("orbital.states must not be empty")
    hint = orbital.get("energy_hint_ev")
    try:
        hint = None if hint is None else float(hint)
        probability = float(viewer.get("enclosed_probability", 0.80))
    except (TypeError, ValueError) as exc:
        raise InputError("invalid Orbital numeric setting") from exc
    representation = str(viewer.get("default_representation", "phase"))
    if hint is not None and not np.isfinite(hint):
        raise InputError("orbital.energy_hint_ev must be finite")
    if representation not in {"phase", "density"}:
        raise InputError("viewer.default_representation must be phase or density")
    if not 0.50 <= probability <= 0.99:
        raise InputError("viewer.enclosed_probability must be 0.50 through 0.99")
    return OrbitalConfig(
        source,
        _path(base, input_section.get("bundle"), "input.bundle"),
        states.strip(),
        grid,
        hint,
        _path(base, basis.get("definition"), "basis.definition"),
        device,
        chunk_points,
        representation,
        probability,
        _path(base, output.get("directory"), "output.directory"),
    )


def _cuda_token(index: int, inherited: str | None) -> str:
    if inherited:
        visible = [value.strip() for value in inherited.split(",") if value.strip()]
        if index >= len(visible):
            raise InputError(
                f"GPU index {index} exceeds CUDA_VISIBLE_DEVICES={inherited}"
            )
        return visible[index]
    return str(index)


def _require_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise InputError(f"output directory is not empty: {path}")


def _prepare_work(
    path: Path,
    specification: dict[str, Any],
    *,
    resume: bool,
) -> None:
    run_path = path / "run.json"
    if resume:
        try:
            observed = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InputError(f"cannot resume work directory {path}") from exc
        if observed != specification:
            raise InputError("resume work belongs to another calculation")
        return
    if path.exists():
        raise InputError(f"work directory already exists: {path}; use --resume")
    path.mkdir(parents=True)
    run_path.write_text(
        json.dumps(specification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _publish_directory(output: Path, publish: Path, work: Path) -> None:
    if output.exists():
        output.rmdir()
    os.replace(publish, output)
    shutil.rmtree(work)


def _band_specification(
    config: BandConfig,
    bundle: LocalizedOperatorBundle,
) -> dict[str, Any]:
    return {
        "kind": "band",
        "bundle": str(config.bundle),
        "basis_identity": bundle.basis.basis_identity,
        "n_orbitals": bundle.n_orbitals,
        "interval_ev": [
            config.interval.lower_ev,
            config.interval.upper_ev,
        ],
        "reference": [
            config.reference_mode,
            config.reference_value_ev,
        ],
        "kpoints": config.kpath.fractional_points.tolist(),
        "nodes": config.kpath.node_indices.tolist(),
        "labels": list(config.kpath.node_labels),
        "solver": asdict(config.settings),
    }


def _band_worker(
    config: BandConfig,
    reference_energy_ev: float,
    rank: int,
    world_size: int,
    cuda_token: str,
) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_token
    bundle = load_bundle(config.bundle)
    calculation = BandCalculation(
        bundle,
        config.kpath,
        config.interval,
        reference_energy_ev,
        config.settings,
    )
    completed = 0
    try:
        for index in range(rank, config.kpath.n_points, world_size):
            path = config.work / f"k-{index:06d}.npz"
            point = read_point(
                path,
                index=index,
                k_fractional=config.kpath.fractional_points[index],
            )
            if point is None:
                print(
                    f"[gpu {rank}] k point {index + 1}/{config.kpath.n_points}",
                    flush=True,
                )
                point = calculation.solve_point(index)
                write_point(path, point)
            completed += 1
    finally:
        calculation.close()
    return completed


def run_band(path: str | Path, *, resume: bool = False) -> Path:
    config = read_band_config(path)
    _require_output(config.output)
    bundle = load_bundle(config.bundle)
    reference = resolve_energy_reference(
        bundle,
        config.reference_mode,
        config.reference_value_ev,
    )
    _prepare_work(
        config.work,
        _band_specification(config, bundle),
        resume=resume,
    )
    started = perf_counter()
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = tuple(_cuda_token(index, inherited) for index in config.devices)
    if len(tokens) == 1:
        _band_worker(config, reference, 0, 1, tokens[0])
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(tokens),
            mp_context=context,
        ) as pool:
            futures = [
                pool.submit(
                    _band_worker,
                    config,
                    reference,
                    rank,
                    len(tokens),
                    token,
                )
                for rank, token in enumerate(tokens)
            ]
            for future in futures:
                future.result()
    points = []
    for index, kpoint in enumerate(config.kpath.fractional_points):
        point = read_point(
            config.work / f"k-{index:06d}.npz",
            index=index,
            k_fractional=kpoint,
        )
        if point is None:
            raise SolverError(f"Band point {index} is missing")
        points.append(point)
    calculation = BandCalculation(
        bundle,
        config.kpath,
        config.interval,
        reference,
        config.settings,
    )
    try:
        result = calculation.build_result(
            points,
            elapsed_seconds=perf_counter() - started,
        )
    finally:
        calculation.close()
    publish = config.output.with_name(f".{config.output.name}.publish")
    if publish.exists():
        shutil.rmtree(publish)
    write_band_output(result, publish)
    _publish_directory(config.output, publish, config.work)
    print(config.output)
    return config.output


def _openmx_provider(
    path: Path,
    expected_identity: str,
) -> OpenMXPAOProvider:
    source, values = _toml(path)
    identity = values.get("basis_identity")
    if identity != expected_identity:
        raise InputError(
            "OpenMX basis_identity differs from the operator bundle"
        )
    files = values.get("pao_files")
    if not isinstance(files, dict) or not files:
        raise InputError("OpenMX definition needs [pao_files]")
    paos = {}
    for atomic_number, value in files.items():
        try:
            number = int(atomic_number)
        except (TypeError, ValueError) as exc:
            raise InputError("PAO keys must be atomic numbers") from exc
        paos[number] = read_openmx_pao(
            _path(source.parent, value, f"pao_files.{atomic_number}")
        )
    return OpenMXPAOProvider(expected_identity, paos)


def _orbital_specification(
    config: OrbitalConfig,
    bundle: LocalizedOperatorBundle,
) -> dict[str, Any]:
    return {
        "kind": "orbital",
        "bundle": str(config.bundle),
        "basis_identity": bundle.basis.basis_identity,
        "n_orbitals": bundle.n_orbitals,
        "states": config.states,
        "grid": {
            "shape": config.grid.shape,
            "spacing_angstrom": config.grid.spacing_angstrom,
        },
        "energy_hint_ev": config.energy_hint_ev,
        "basis_definition": str(config.basis_definition),
        "chunk_points": config.chunk_points,
    }


def run_orbital(path: str | Path, *, resume: bool = False) -> Path:
    config = read_orbital_config(path)
    _require_output(config.output)
    bundle = load_bundle(config.bundle)
    provider = _openmx_provider(
        config.basis_definition,
        bundle.basis.basis_identity,
    )
    _prepare_work(
        config.work,
        _orbital_specification(config, bundle),
        resume=resume,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_token(
        config.device,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    calculation = OrbitalCalculation(
        bundle,
        config.states,
        config.grid,
        config.energy_hint_ev,
    )
    try:
        eigensystem_path = config.work / "eigensystem.npz"
        eigensystem = read_eigensystem(eigensystem_path)
        if eigensystem is None:
            print("[orbital] solving selected states", flush=True)
            eigensystem = calculation.solve_eigensystem()
            write_eigensystem(eigensystem_path, eigensystem)
        resources = FieldResources("cuda:0", config.chunk_points)
        for column, state_number in enumerate(eigensystem.state_numbers):
            field_path = config.work / f"state-{int(state_number):06d}.npz"
            state = read_state_field(field_path)
            if state is None:
                print(
                    f"[orbital] field {column + 1}/"
                    f"{eigensystem.state_numbers.size}",
                    flush=True,
                )
                started = perf_counter()
                state = calculation.evaluate_state(
                    eigensystem,
                    provider,
                    resources,
                    column,
                )
                state.evaluation_seconds = perf_counter() - started
                write_state_field(field_path, state)
    finally:
        calculation.close()

    def fields():
        for state_number in eigensystem.state_numbers:
            state = read_state_field(
                config.work / f"state-{int(state_number):06d}.npz"
            )
            if state is None:
                raise SolverError(f"Orbital state {state_number} is missing")
            yield state

    publish = config.output.with_name(f".{config.output.name}.publish")
    if publish.exists():
        shutil.rmtree(publish)
    write_orbital_viewer(
        publish,
        bundle.structure,
        eigensystem,
        fields(),
        default_probability=config.enclosed_probability,
        default_representation=config.default_representation,
    )
    _publish_directory(config.output, publish, config.work)
    print(config.output)
    return config.output


def serve(path: str | Path, *, host: str, port: int) -> None:
    root = Path(path).expanduser().resolve()
    if not (root / "index.html").is_file():
        raise InputError(f"no index.html in {root}")
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    try:
        with ThreadingHTTPServer((host, port), handler) as server:
            print(f"http://{host}:{port}/", flush=True)
            server.serve_forever()
    except OSError as exc:
        raise InputError(f"cannot serve {root}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aospectrum",
        description="Sparse Band and Orbital tools for localized AO Hamiltonians.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("band", "calculate an energy-window band structure"),
        ("orbital", "calculate and render selected Gamma-point orbitals"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", type=Path)
        command.add_argument("--resume", action="store_true")
    view = commands.add_parser("view", help="serve an Orbital viewer")
    view.add_argument("output", type=Path)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "band":
            run_band(arguments.config, resume=arguments.resume)
        elif arguments.command == "orbital":
            run_orbital(arguments.config, resume=arguments.resume)
        else:
            serve(
                arguments.output,
                host=arguments.host,
                port=arguments.port,
            )
    except AOSpectrumError as exc:
        print(f"aospectrum: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

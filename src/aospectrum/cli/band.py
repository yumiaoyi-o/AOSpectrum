"""Configured Band application with deterministic device sharding."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import multiprocessing
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
from uuid import uuid4

from aospectrum import __version__, load_bundle
from aospectrum.band import (
    BandCalculator,
    BandExecutor,
    BandRequest,
    EnergyReference,
    merge_band_journals,
    plot_band_result,
    read_kpath,
    write_band_result,
)
from aospectrum.cli.config import (
    ConfigDocument,
    cuda_visible_token,
    load_config,
    reject_unknown_keys,
    required_number_pair,
    required_string,
)
from aospectrum.errors import ArtifactError, ContractError
from aospectrum.model.spectra import EnergyInterval
from aospectrum.solvers import create_band_solver


@dataclass(frozen=True, slots=True)
class BandRunConfiguration:
    source: Path
    bundle: Path
    request: BandRequest
    devices: tuple[str, ...]
    contour_batch_size: int
    output: Path

    @property
    def checkpoint_root(self) -> Path:
        return self.output.with_name(f".{self.output.name}.checkpoints")


def read_band_configuration(path: str | Path) -> BandRunConfiguration:
    document = load_config(path)
    document.require_sections({"input", "band", "compute", "resources", "output"})
    input_section = document.section("input")
    band_section = document.section("band")
    compute_section = document.section("compute")
    resources_section = document.section("resources")
    output_section = document.section("output")
    reject_unknown_keys(input_section, {"bundle"}, context="input")
    reject_unknown_keys(
        band_section,
        {
            "energy_interval_ev",
            "energy_reference",
            "energy_reference_ev",
            "kpath",
        },
        context="band",
    )
    reject_unknown_keys(compute_section, {"devices"}, context="compute")
    reject_unknown_keys(
        resources_section,
        {"contour_batch_size"},
        context="resources",
    )
    reject_unknown_keys(output_section, {"directory"}, context="output")
    interval = required_number_pair(
        band_section,
        "energy_interval_ev",
        context="band",
    )
    reference = _energy_reference(band_section)
    raw_devices = compute_section.get("devices")
    if not isinstance(raw_devices, list) or not raw_devices:
        raise ContractError(
            "compute.devices must contain one or more GPU indices"
        )
    devices = tuple(_compute_device(value) for value in raw_devices)
    if len(set(devices)) != len(devices):
        raise ContractError("compute.devices must not contain duplicates")
    contour_batch = resources_section.get("contour_batch_size", 1)
    if (
        isinstance(contour_batch, bool)
        or not isinstance(contour_batch, int)
        or contour_batch <= 0
    ):
        raise ContractError("resources.contour_batch_size must be positive")
    kpath_path = document.resolve_path(
        required_string(band_section, "kpath", context="band"),
        name="band.kpath",
    )
    return BandRunConfiguration(
        source=document.path,
        bundle=document.resolve_path(
            required_string(input_section, "bundle", context="input"),
            name="input.bundle",
        ),
        request=BandRequest(
            kpath=read_kpath(kpath_path),
            display_interval_ev=EnergyInterval(*interval),
            energy_reference=reference,
            precision="float32",
        ),
        devices=devices,
        contour_batch_size=contour_batch,
        output=document.resolve_path(
            required_string(output_section, "directory", context="output"),
            name="output.directory",
        ),
    )


def _energy_reference(section: Mapping[str, Any]) -> EnergyReference:
    value = required_string(
        section,
        "energy_reference",
        context="band",
    )
    if value in {"input_zero", "absolute"}:
        return EnergyReference("absolute")
    if value == "bundle_fermi":
        return EnergyReference("bundle_fermi")
    if value == "fixed":
        fixed = section.get("energy_reference_ev")
        if isinstance(fixed, bool):
            raise ContractError("band.energy_reference_ev must be numeric")
        try:
            return EnergyReference("fixed", float(fixed))
        except (TypeError, ValueError) as exc:
            raise ContractError(
                "fixed Band reference requires band.energy_reference_ev"
            ) from exc
    raise ContractError(
        "band.energy_reference must be input_zero, bundle_fermi or fixed"
    )


def _compute_device(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("compute.devices entries must be nonnegative GPU indices")
    return f"cuda:{value}"


def _create_calculator(
    configuration: BandRunConfiguration,
) -> BandCalculator:
    solver = create_band_solver(
        contour_batch_size=configuration.contour_batch_size,
    )
    return BandCalculator(solver)


def _run_shard(
    configuration: BandRunConfiguration,
    rank: int,
    world_size: int,
    cuda_token: str | None,
) -> int:
    if cuda_token is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_token
    bundle = load_bundle(configuration.bundle, verify_hashes=False)
    calculator = _create_calculator(configuration)
    try:
        BandExecutor(
            calculator,
            configuration.checkpoint_root / f"rank-{rank:03d}",
            rank=rank,
            world_size=world_size,
        ).run(bundle, configuration.request)
    finally:
        calculator.close()
    return rank


def _write_run_state(
    root: Path,
    *,
    status: str,
    configuration: BandRunConfiguration,
    error: str | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "aospectrum.band-run/v1",
        "status": status,
        "aospectrum_version": __version__,
        "config": str(configuration.source),
        "output": str(configuration.output),
        "world_size": len(configuration.devices),
        "devices": list(configuration.devices),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    temporary = root / f".run-manifest-{uuid4().hex}.json"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, root / "run-manifest.json")


def _require_run_paths(
    configuration: BandRunConfiguration,
    *,
    resume: bool,
) -> None:
    output = configuration.output
    if output.exists():
        if not output.is_dir():
            raise ArtifactError(f"Band output path is not a directory: {output}")
        if any(output.iterdir()):
            raise ArtifactError(f"Band output is not empty: {output}")
    checkpoint = configuration.checkpoint_root
    if resume:
        if not checkpoint.is_dir():
            raise ArtifactError(
                f"Band resume checkpoint does not exist: {checkpoint}"
            )
    elif checkpoint.exists():
        raise ArtifactError(
            f"Band checkpoint already exists: {checkpoint}; "
            "use --resume or choose a new output directory"
        )


def run_band_config(path: str | Path, *, resume: bool = False) -> Path:
    configuration = read_band_configuration(path)
    _require_run_paths(configuration, resume=resume)
    bundle = load_bundle(configuration.bundle)
    _write_run_state(
        configuration.checkpoint_root,
        status="running",
        configuration=configuration,
    )
    world_size = len(configuration.devices)
    inherited_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_tokens = tuple(
        (
            cuda_visible_token(device, inherited_cuda)
            if device.startswith("cuda:")
            else None
        )
        for device in configuration.devices
    )
    try:
        if world_size == 1:
            _run_shard(
                configuration,
                0,
                1,
                cuda_tokens[0],
            )
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=world_size,
                mp_context=context,
            ) as pool:
                futures = [
                    pool.submit(
                        _run_shard,
                        configuration,
                        rank,
                        world_size,
                        cuda_tokens[rank],
                    )
                    for rank in range(world_size)
                ]
                completed = sorted(future.result() for future in futures)
            if completed != list(range(world_size)):
                raise ArtifactError(
                    "Band workers did not return exact rank coverage"
                )
        calculator = _create_calculator(configuration)
        try:
            prepared = calculator.prepare(bundle, configuration.request)
            result = merge_band_journals(
                prepared,
                [
                    configuration.checkpoint_root / f"rank-{rank:03d}"
                    for rank in range(world_size)
                ],
            )
        finally:
            calculator.close()
        temporary_output = configuration.output.with_name(
            f".{configuration.output.name}.tmp-{uuid4().hex}"
        )
        try:
            output = write_band_result(result, temporary_output)
            plot_band_result(result, output / "band.png")
            _write_band_html(output)
            if configuration.output.exists():
                configuration.output.rmdir()
            os.replace(temporary_output, configuration.output)
        except Exception:
            shutil.rmtree(temporary_output, ignore_errors=True)
            raise
    except Exception as exc:
        _write_run_state(
            configuration.checkpoint_root,
            status="failed",
            configuration=configuration,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    shutil.rmtree(configuration.checkpoint_root)
    output = configuration.output
    print(output)
    return output


def _write_band_html(root: Path) -> None:
    title = "AOSpectrum Band"
    (root / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{margin:0;font-family:system-ui,sans-serif;"
        "background:#f5f6f8;color:#171a20}header{padding:14px 20px;"
        "background:#fff;border-bottom:1px solid #d4d8df}h1{font-size:18px;"
        "margin:0}main{max-width:1100px;margin:auto;padding:18px}"
        "img{display:block;width:100%;height:auto;background:#fff}"
        "nav{margin-top:12px;display:flex;gap:16px}a{color:#2457a7}</style>"
        "</head><body><header><h1>AOSpectrum Band</h1></header><main>"
        '<img src="band.png" alt="Band structure">'
        '<nav><a href="bands.csv">Band table</a>'
        '<a href="quality.json">Numerical quality</a>'
        '<a href="receipt.json">Resource receipt</a></nav>'
        "</main></body></html>\n",
        encoding="utf-8",
    )

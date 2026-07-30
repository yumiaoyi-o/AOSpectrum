"""The two production numerical backends owned by AOSpectrum."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec

from aospectrum.errors import BackendUnavailableError
from aospectrum.solvers.indexed import IndexedStateSolver
from aospectrum.solvers.interval import EnergyIntervalSolver


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    name: str
    application: str
    precisions: tuple[str, ...]
    dependency: str
    installed: bool
    notes: str


def _cuda_extension_installed() -> bool:
    try:
        return find_spec("aospectrum._aospectrum_cuda") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def backend_descriptors() -> tuple[BackendDescriptor, ...]:
    native = _cuda_extension_installed()
    return (
        BackendDescriptor(
            name="ciss-cudss",
            application="band",
            precisions=("complex64",),
            dependency="AOSpectrum CUDA extension and cuDSS",
            installed=native,
            notes="CISS energy-interval solve on one CUDA device",
        ),
        BackendDescriptor(
            name="primme-cudss",
            application="orbital",
            precisions=("real32-gamma",),
            dependency="AOSpectrum CUDA extension, PRIMME and cuDSS",
            installed=native,
            notes="Gamma-point indexed states with inertia certification",
        ),
    )


def _require_backend(application: str) -> BackendDescriptor:
    for descriptor in backend_descriptors():
        if descriptor.application == application:
            if not descriptor.installed:
                raise BackendUnavailableError(
                    f"{application} requires {descriptor.dependency}"
                )
            return descriptor
    raise AssertionError(f"no production backend registered for {application}")


def create_band_solver(
    *,
    contour_batch_size: int = 1,
) -> EnergyIntervalSolver:
    _require_backend("band")
    from aospectrum.solvers.backends.cudss_ciss import (
        CissSettings,
        CudssCissSolver,
    )

    return CudssCissSolver(
        precision="float32",
        settings=CissSettings(
            contour_batch_size=contour_batch_size,
        ),
    )


def create_orbital_solver() -> IndexedStateSolver:
    _require_backend("orbital")
    from aospectrum.solvers.backends.primme_cudss import PrimmeCudssSolver

    return PrimmeCudssSolver(precision="float32")

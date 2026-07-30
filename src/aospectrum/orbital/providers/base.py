"""Real-space basis-provider protocol and field result."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from aospectrum.errors import ContractError
from aospectrum.model.basis import OrbitalBasisLayout
from aospectrum.model.structure import AtomicStructure
from aospectrum.orbital.model import GridSpec
from aospectrum.orbital.volume import OrbitalField


@dataclass(frozen=True, slots=True)
class FieldResources:
    device: str = "cpu"
    chunk_points: int = 65_536

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device.strip():
            raise ContractError("field device must not be empty")
        if (
            isinstance(self.chunk_points, bool)
            or int(self.chunk_points) <= 0
        ):
            raise ContractError("field chunk_points must be positive")
        object.__setattr__(self, "device", self.device.strip())
        object.__setattr__(self, "chunk_points", int(self.chunk_points))


@runtime_checkable
class BasisFunctionProvider(Protocol):
    basis_identity: str

    def evaluate_orbital(
        self,
        structure: AtomicStructure,
        layout: OrbitalBasisLayout,
        coefficients: np.ndarray,
        grid: GridSpec,
        resources: FieldResources,
        *,
        precision: str,
    ) -> OrbitalField: ...

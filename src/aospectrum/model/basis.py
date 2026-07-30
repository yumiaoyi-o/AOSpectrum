"""Global AO ordering and per-atom layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from aospectrum.errors import ContractError

from ._arrays import frozen_array


@dataclass(frozen=True, slots=True, eq=False)
class OrbitalBasisLayout:
    """The unique coefficient ordering shared by H, S, eigenvectors and fields."""

    n_atoms: int
    orbital_atom: np.ndarray
    local_index: np.ndarray
    basis_family: str
    basis_identity: str
    orbital_labels: tuple[str, ...] = ()
    angular_momentum: np.ndarray | None = None
    radial_index: np.ndarray | None = None
    real_harmonic_index: np.ndarray | None = None
    spinor_width: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _atom_offsets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            isinstance(self.n_atoms, bool)
            or int(self.n_atoms) <= 0
        ):
            raise ContractError("basis n_atoms must be a positive integer")
        orbital_atom = frozen_array(
            self.orbital_atom,
            name="orbital_atom",
            ndim=1,
            dtype=np.int32,
            kinds="iu",
        )
        local_index = frozen_array(
            self.local_index,
            name="local_index",
            ndim=1,
            dtype=np.int32,
            kinds="iu",
        )
        if orbital_atom.size == 0 or orbital_atom.shape != local_index.shape:
            raise ContractError(
                "orbital_atom and local_index must be equal nonempty arrays"
            )
        if np.any(orbital_atom < 0) or np.any(orbital_atom >= int(self.n_atoms)):
            raise ContractError("orbital_atom contains an out-of-range atom")
        if np.any(local_index < 0):
            raise ContractError("local_index must be nonnegative")
        if np.any(np.diff(orbital_atom) < 0):
            raise ContractError("global AO order must be grouped by atom")
        counts = np.bincount(orbital_atom, minlength=int(self.n_atoms))
        if np.any(counts == 0):
            raise ContractError("each atom must own at least one AO")
        offsets = np.zeros(int(self.n_atoms) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts, dtype=np.int64)
        for atom in range(int(self.n_atoms)):
            start, stop = int(offsets[atom]), int(offsets[atom + 1])
            expected = np.arange(stop - start, dtype=np.int32)
            if not np.array_equal(local_index[start:stop], expected):
                raise ContractError(
                    f"local_index for atom {atom} must be contiguous from zero"
                )
        for name, value in (
            ("basis_family", self.basis_family),
            ("basis_identity", self.basis_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{name} must be nonempty")
        labels = tuple(self.orbital_labels)
        if labels and (
            len(labels) != orbital_atom.size
            or not all(isinstance(value, str) and value for value in labels)
        ):
            raise ContractError(
                "orbital_labels must be empty or contain one label per AO"
            )
        descriptor_values = (
            self.angular_momentum,
            self.radial_index,
            self.real_harmonic_index,
        )
        if any(value is not None for value in descriptor_values):
            if not all(value is not None for value in descriptor_values):
                raise ContractError(
                    "real-space AO descriptors must be provided together"
                )
            angular = frozen_array(
                self.angular_momentum,
                name="angular_momentum",
                ndim=1,
                dtype=np.int32,
                kinds="iu",
            )
            radial = frozen_array(
                self.radial_index,
                name="radial_index",
                ndim=1,
                dtype=np.int32,
                kinds="iu",
            )
            harmonic = frozen_array(
                self.real_harmonic_index,
                name="real_harmonic_index",
                ndim=1,
                dtype=np.int32,
                kinds="iu",
            )
            if not (
                angular.shape
                == radial.shape
                == harmonic.shape
                == orbital_atom.shape
            ):
                raise ContractError(
                    "real-space AO descriptors must contain one value per AO"
                )
            if (
                np.any(angular < 0)
                or np.any(radial < 0)
                or np.any(harmonic < 0)
                or np.any(harmonic > 2 * angular)
            ):
                raise ContractError("real-space AO descriptors are invalid")
        else:
            angular = radial = harmonic = None
        if (
            isinstance(self.spinor_width, bool)
            or int(self.spinor_width) <= 0
        ):
            raise ContractError("spinor_width must be a positive integer")
        offsets.setflags(write=False)
        object.__setattr__(self, "n_atoms", int(self.n_atoms))
        object.__setattr__(self, "orbital_atom", orbital_atom)
        object.__setattr__(self, "local_index", local_index)
        object.__setattr__(self, "orbital_labels", labels)
        object.__setattr__(self, "angular_momentum", angular)
        object.__setattr__(self, "radial_index", radial)
        object.__setattr__(self, "real_harmonic_index", harmonic)
        object.__setattr__(self, "spinor_width", int(self.spinor_width))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "_atom_offsets", offsets)

    @property
    def n_orbitals(self) -> int:
        return int(self.orbital_atom.size)

    @property
    def atom_offsets(self) -> np.ndarray:
        return self._atom_offsets

    @property
    def has_real_space_descriptors(self) -> bool:
        return self.angular_momentum is not None

    def orbital_count(self, atom: int) -> int:
        if isinstance(atom, bool) or not 0 <= int(atom) < self.n_atoms:
            raise ContractError("atom index is out of range")
        return int(self._atom_offsets[int(atom) + 1] - self._atom_offsets[int(atom)])

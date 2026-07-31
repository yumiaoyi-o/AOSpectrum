"""Minimal OpenMX ``.pao`` reader used by the orbital provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .errors import InputError


@dataclass(frozen=True, slots=True)
class OpenMXPAOData:
    basis_name: str
    cutoff_bohr: float
    radial_mesh_bohr: np.ndarray
    radial_functions: dict[int, np.ndarray]

    def radial(self, angular_momentum: int, radial_index: int) -> np.ndarray:
        try:
            return self.radial_functions[angular_momentum][radial_index]
        except (KeyError, IndexError) as exc:
            raise InputError(
                f"PAO {self.basis_name} has no l={angular_momentum}, "
                f"radial={radial_index}"
            ) from exc


def _number(text: str, keyword: str, *, integer: bool = False) -> float | int:
    pattern = rf"{re.escape(keyword)}\s+([\d.eE+-]+)"
    match = re.search(pattern, text)
    if match is None:
        raise InputError(f"OpenMX PAO is missing {keyword}")
    return int(match.group(1)) if integer else float(match.group(1))


def read_openmx_pao(path: str | Path) -> OpenMXPAOData:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InputError(f"cannot read OpenMX PAO {source}: {exc}") from exc
    cutoff = float(_number(text, "radial.cutoff.pao"))
    lmax = int(_number(text, "PAO.Lmax", integer=True))
    multiplicity = int(_number(text, "PAO.Mul", integer=True))
    mesh: np.ndarray | None = None
    radial: dict[int, np.ndarray] = {}
    for angular in range(lmax + 1):
        start_tag = f"<pseudo.atomic.orbitals.L={angular}"
        end_tag = f"pseudo.atomic.orbitals.L={angular}>"
        start = text.find(start_tag)
        if start < 0:
            raise InputError(f"PAO {source} is missing l={angular} block")
        try:
            start = text.index("\n", start) + 1
        except ValueError as exc:
            raise InputError(
                f"PAO {source} has a malformed l={angular} tag"
            ) from exc
        stop = text.find(end_tag, start)
        if stop < 0:
            raise InputError(f"PAO {source} has no l={angular} closing tag")
        try:
            values = np.asarray(
                [float(value) for value in text[start:stop].split()],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise InputError(
                f"PAO {source} l={angular} contains non-numeric data"
            ) from exc
        columns = 2 + multiplicity
        if values.size == 0 or values.size % columns:
            raise InputError(f"PAO {source} l={angular} block shape is invalid")
        table = values.reshape(-1, columns)
        current_mesh = table[:, 1]
        if mesh is None:
            mesh = current_mesh
        elif not np.array_equal(mesh, current_mesh):
            raise InputError(f"PAO {source} radial meshes differ by l")
        radial[angular] = np.ascontiguousarray(table[:, 2:].T)
    assert mesh is not None
    if cutoff <= 0.0 or np.any(np.diff(mesh) <= 0.0):
        raise InputError(f"PAO {source} has an invalid cutoff or radial mesh")
    mesh.setflags(write=False)
    for values in radial.values():
        values.setflags(write=False)
    return OpenMXPAOData(source.stem, cutoff, mesh, radial)

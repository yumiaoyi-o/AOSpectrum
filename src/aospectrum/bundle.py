"""Read and write the fixed AOSpectrum operator-bundle layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import InputError
from .data import (
    AtomicStructure,
    ElectronicFilling,
    LocalizedOperatorBlocks,
    LocalizedOperatorBundle,
    OrbitalBasisLayout,
)


SCHEMA = "aospectrum.operator-bundle/v1"


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InputError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_bundle(
    root: str | Path,
    *,
    mmap_mode: str | None = "r",
) -> LocalizedOperatorBundle:
    """Load the fixed directory layout without checksumming large arrays."""

    root = Path(root).expanduser().resolve()
    manifest = _json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise InputError(f"unsupported operator bundle: {manifest.get('schema')}")
    layout = _json(root / "basis" / "layout.json")

    def array(path: str) -> np.ndarray:
        try:
            return np.load(
                root / path,
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )
        except (OSError, ValueError) as exc:
            raise InputError(f"cannot read bundle array {path}: {exc}") from exc

    filling = None
    filling_path = root / "filling.json"
    if filling_path.is_file():
        raw = _json(filling_path)
        filling = ElectronicFilling(
            electron_count=raw["electron_count"],
            spin_degeneracy=raw.get("spin_degeneracy", 2),
            fermi_energy_ev=raw.get("fermi_energy_ev"),
        )
    provenance_path = root / "provenance.json"
    provenance = _json(provenance_path) if provenance_path.is_file() else {}
    descriptors = bool(layout.get("has_real_space_descriptors", False))
    return LocalizedOperatorBundle(
        structure=AtomicStructure(
            cell_angstrom=array("structure/cell.npy"),
            positions_angstrom=array("structure/positions.npy"),
            atomic_numbers=array("structure/atomic_numbers.npy"),
            periodic=tuple(manifest.get("periodic", (True, True, True))),
            label=manifest.get("structure_label"),
        ),
        basis=OrbitalBasisLayout(
            n_atoms=layout["n_atoms"],
            orbital_atom=array("basis/orbital_atom.npy"),
            local_index=array("basis/local_index.npy"),
            basis_family=str(layout["basis_family"]),
            basis_identity=str(layout["basis_identity"]),
            orbital_labels=tuple(layout.get("orbital_labels", ())),
            angular_momentum=(
                array("basis/angular_momentum.npy") if descriptors else None
            ),
            radial_index=(
                array("basis/radial_index.npy") if descriptors else None
            ),
            real_harmonic_index=(
                array("basis/real_harmonic_index.npy") if descriptors else None
            ),
            spinor_width=layout.get("spinor_width", 1),
            metadata=dict(layout.get("metadata", {})),
        ),
        operators=LocalizedOperatorBlocks(
            row_atom=array("operators/row_atom.npy"),
            col_atom=array("operators/col_atom.npy"),
            lattice_shift=array("operators/lattice_shift.npy"),
            block_offsets=array("operators/block_offsets.npy"),
            block_shapes=array("operators/block_shapes.npy"),
            h_values=array("operators/h_values.npy"),
            s_values=array("operators/s_values.npy"),
        ),
        filling=filling,
        provenance=provenance,
    )


def write_bundle(
    bundle: LocalizedOperatorBundle,
    root: str | Path,
) -> Path:
    """Write one transparent bundle; the destination must be empty."""

    root = Path(root).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise InputError(f"bundle destination is not empty: {root}")
    for name in ("structure", "basis", "operators"):
        (root / name).mkdir(parents=True, exist_ok=True)

    arrays = {
        "structure/cell.npy": bundle.structure.cell_angstrom,
        "structure/positions.npy": bundle.structure.positions_angstrom,
        "structure/atomic_numbers.npy": bundle.structure.atomic_numbers,
        "basis/orbital_atom.npy": bundle.basis.orbital_atom,
        "basis/local_index.npy": bundle.basis.local_index,
        "operators/row_atom.npy": bundle.operators.row_atom,
        "operators/col_atom.npy": bundle.operators.col_atom,
        "operators/lattice_shift.npy": bundle.operators.lattice_shift,
        "operators/block_offsets.npy": bundle.operators.block_offsets,
        "operators/block_shapes.npy": bundle.operators.block_shapes,
        "operators/h_values.npy": bundle.operators.h_values,
        "operators/s_values.npy": bundle.operators.s_values,
    }
    if bundle.basis.has_real_space_descriptors:
        arrays.update(
            {
                "basis/angular_momentum.npy": bundle.basis.angular_momentum,
                "basis/radial_index.npy": bundle.basis.radial_index,
                "basis/real_harmonic_index.npy": (
                    bundle.basis.real_harmonic_index
                ),
            }
        )
    for relative, values in arrays.items():
        np.save(root / relative, values, allow_pickle=False)

    _write_json(
        root / "manifest.json",
        {
            "schema": SCHEMA,
            "periodic": list(bundle.structure.periodic),
            "structure_label": bundle.structure.label,
            "hamiltonian_unit": "eV",
            "length_unit": "angstrom",
        },
    )
    _write_json(
        root / "basis" / "layout.json",
        {
            "n_atoms": bundle.basis.n_atoms,
            "basis_family": bundle.basis.basis_family,
            "basis_identity": bundle.basis.basis_identity,
            "orbital_labels": list(bundle.basis.orbital_labels),
            "has_real_space_descriptors": (
                bundle.basis.has_real_space_descriptors
            ),
            "spinor_width": bundle.basis.spinor_width,
            "metadata": bundle.basis.metadata,
        },
    )
    if bundle.filling is not None:
        _write_json(
            root / "filling.json",
            {
                "mode": "closed_shell",
                "electron_count": bundle.filling.electron_count,
                "spin_degeneracy": bundle.filling.spin_degeneracy,
                "fermi_energy_ev": bundle.filling.fermi_energy_ev,
            },
        )
    _write_json(root / "provenance.json", bundle.provenance)
    return root

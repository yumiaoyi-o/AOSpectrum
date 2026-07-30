"""Directory-based AOSpectrum operator bundle I/O."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aospectrum.errors import BundleIOError, ContractError
from aospectrum.model.basis import OrbitalBasisLayout
from aospectrum.model.operators import (
    ElectronicFilling,
    LocalizedOperatorBlocks,
    LocalizedOperatorBundle,
)
from aospectrum.model.structure import AtomicStructure


SCHEMA = "aospectrum.operator-bundle/v1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleIOError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleIOError(f"JSON file must contain an object: {path}")
    return value


def _array_record(relative_path: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": relative_path,
        "dtype": np.dtype(array.dtype).name,
        "shape": [int(value) for value in array.shape],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_bundle(
    bundle: LocalizedOperatorBundle,
    root: str | Path,
) -> Path:
    """Write one transparent, memory-mappable operator bundle directory."""

    if not isinstance(bundle, LocalizedOperatorBundle):
        raise BundleIOError("bundle must be LocalizedOperatorBundle")
    destination = Path(root)
    if destination.exists():
        if not destination.is_dir():
            raise BundleIOError(
                f"bundle destination is not a directory: {destination}"
            )
        if any(destination.iterdir()):
            raise BundleIOError(
                f"bundle destination is not empty: {destination}"
            )
    try:
        (destination / "structure").mkdir(parents=True, exist_ok=True)
        (destination / "basis").mkdir(parents=True, exist_ok=True)
        (destination / "operators").mkdir(parents=True, exist_ok=True)

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
            assert bundle.basis.angular_momentum is not None
            assert bundle.basis.radial_index is not None
            assert bundle.basis.real_harmonic_index is not None
            arrays.update(
                {
                    "basis/angular_momentum.npy": (
                        bundle.basis.angular_momentum
                    ),
                    "basis/radial_index.npy": bundle.basis.radial_index,
                    "basis/real_harmonic_index.npy": (
                        bundle.basis.real_harmonic_index
                    ),
                }
            )
        for relative_path, array in arrays.items():
            np.save(destination / relative_path, array, allow_pickle=False)

        _write_json(
            destination / "basis" / "layout.json",
            {
                "n_atoms": bundle.basis.n_atoms,
                "basis_family": bundle.basis.basis_family,
                "basis_identity": bundle.basis.basis_identity,
                "orbital_labels": list(bundle.basis.orbital_labels),
                "has_real_space_descriptors": (
                    bundle.basis.has_real_space_descriptors
                ),
                "spinor_width": bundle.basis.spinor_width,
                "metadata": dict(bundle.basis.metadata),
            },
        )
        if bundle.filling is not None:
            _write_json(
                destination / "filling.json",
                {
                    "mode": bundle.filling.mode,
                    "electron_count": bundle.filling.electron_count,
                    "spin_degeneracy": bundle.filling.spin_degeneracy,
                    "fermi_energy_ev": bundle.filling.fermi_energy_ev,
                },
            )
        _write_json(destination / "provenance.json", dict(bundle.provenance))
        _write_json(
            destination / "manifest.json",
            {
                "schema": SCHEMA,
                "hamiltonian_unit": "eV",
                "overlap_unit": "dimensionless",
                "length_unit": "angstrom",
                "n_atoms": bundle.structure.n_atoms,
                "n_orbitals": bundle.n_orbitals,
                "n_blocks": bundle.operators.n_blocks,
                "periodic": list(bundle.structure.periodic),
                "structure_label": bundle.structure.label,
                "arrays": {
                    name: {
                        **_array_record(name, array),
                        "sha256": _sha256(destination / name),
                    }
                    for name, array in arrays.items()
                },
            },
        )
    except (OSError, TypeError, ValueError) as exc:
        raise BundleIOError(f"cannot write operator bundle {destination}: {exc}") from exc
    return destination


def _load_array(
    root: Path,
    record: Mapping[str, Any],
    *,
    mmap_mode: str | None,
    verify_hash: bool,
) -> np.ndarray:
    try:
        relative = str(record["path"])
        expected_dtype = np.dtype(str(record["dtype"]))
        expected_shape = tuple(int(value) for value in record["shape"])
        expected_sha256 = str(record["sha256"])
        path = root / relative
        if verify_hash and _sha256(path) != expected_sha256:
            raise BundleIOError(f"bundle array checksum mismatch: {relative}")
        array = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise BundleIOError(f"cannot load bundle array record: {record}") from exc
    if array.dtype != expected_dtype or array.shape != expected_shape:
        raise BundleIOError(
            f"bundle array metadata mismatch for {relative}: "
            f"expected {expected_dtype}{expected_shape}, "
            f"got {array.dtype}{array.shape}"
        )
    return array


def load_bundle(
    root: str | Path,
    *,
    mmap_mode: str | None = "r",
    verify_hashes: bool = True,
) -> LocalizedOperatorBundle:
    """Load a versioned bundle while retaining read-only memory maps."""

    source = Path(root)
    if type(verify_hashes) is not bool:
        raise BundleIOError("verify_hashes must be bool")
    manifest = _load_json(source / "manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise BundleIOError(
            f"unsupported bundle schema: {manifest.get('schema')!r}"
        )
    if manifest.get("hamiltonian_unit") != "eV":
        raise BundleIOError("bundle Hamiltonian unit must be eV")
    if manifest.get("overlap_unit") != "dimensionless":
        raise BundleIOError("bundle overlap unit must be dimensionless")
    if manifest.get("length_unit") != "angstrom":
        raise BundleIOError("bundle length unit must be angstrom")
    array_records = manifest.get("arrays")
    if not isinstance(array_records, dict):
        raise BundleIOError("bundle manifest arrays must be an object")

    def array(name: str) -> np.ndarray:
        record = array_records.get(name)
        if not isinstance(record, dict):
            raise BundleIOError(f"bundle manifest is missing array {name}")
        return _load_array(
            source,
            record,
            mmap_mode=mmap_mode,
            verify_hash=verify_hashes,
        )

    layout = _load_json(source / "basis" / "layout.json")
    filling_path = source / "filling.json"
    filling = None
    if filling_path.exists():
        raw_filling = _load_json(filling_path)
        filling = ElectronicFilling(
            mode=str(raw_filling["mode"]),
            electron_count=raw_filling["electron_count"],
            spin_degeneracy=raw_filling.get("spin_degeneracy", 2),
            fermi_energy_ev=raw_filling.get("fermi_energy_ev"),
        )
    provenance_path = source / "provenance.json"
    provenance = _load_json(provenance_path) if provenance_path.exists() else {}
    provenance = {
        **provenance,
        "bundle_manifest_sha256": _sha256(source / "manifest.json"),
    }
    try:
        structure = AtomicStructure(
            cell_angstrom=array("structure/cell.npy"),
            positions_angstrom=array("structure/positions.npy"),
            atomic_numbers=array("structure/atomic_numbers.npy"),
            periodic=tuple(bool(value) for value in manifest["periodic"]),
            label=manifest.get("structure_label"),
        )
        basis = OrbitalBasisLayout(
            n_atoms=int(layout["n_atoms"]),
            orbital_atom=array("basis/orbital_atom.npy"),
            local_index=array("basis/local_index.npy"),
            basis_family=str(layout["basis_family"]),
            basis_identity=str(layout["basis_identity"]),
            orbital_labels=tuple(str(value) for value in layout["orbital_labels"]),
            angular_momentum=(
                array("basis/angular_momentum.npy")
                if layout.get("has_real_space_descriptors", False)
                else None
            ),
            radial_index=(
                array("basis/radial_index.npy")
                if layout.get("has_real_space_descriptors", False)
                else None
            ),
            real_harmonic_index=(
                array("basis/real_harmonic_index.npy")
                if layout.get("has_real_space_descriptors", False)
                else None
            ),
            spinor_width=int(layout.get("spinor_width", 1)),
            metadata=layout.get("metadata", {}),
        )
        operators = LocalizedOperatorBlocks(
            row_atom=array("operators/row_atom.npy"),
            col_atom=array("operators/col_atom.npy"),
            lattice_shift=array("operators/lattice_shift.npy"),
            block_offsets=array("operators/block_offsets.npy"),
            block_shapes=array("operators/block_shapes.npy"),
            h_values=array("operators/h_values.npy"),
            s_values=array("operators/s_values.npy"),
        )
        return LocalizedOperatorBundle(
            structure=structure,
            basis=basis,
            operators=operators,
            filling=filling,
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        raise BundleIOError(f"invalid operator bundle {source}: {exc}") from exc

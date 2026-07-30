"""Read-only bundle inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aospectrum import load_bundle
from aospectrum.solvers import backend_descriptors


def inspect_bundle(path: str | Path) -> dict[str, object]:
    bundle = load_bundle(path)
    filling = bundle.filling
    block_entries = int(
        np.sum(
            bundle.operators.block_shapes[:, 0].astype(np.int64)
            * bundle.operators.block_shapes[:, 1].astype(np.int64)
        )
    )
    return {
        "schema": "aospectrum.inspect/v1",
        "structure": {
            "label": bundle.structure.label,
            "n_atoms": bundle.structure.n_atoms,
            "periodic": list(bundle.structure.periodic),
        },
        "basis": {
            "family": bundle.basis.basis_family,
            "identity": bundle.basis.basis_identity,
            "n_orbitals": bundle.basis.n_orbitals,
            "spinor_width": bundle.basis.spinor_width,
            "has_real_space_descriptors": (
                bundle.basis.has_real_space_descriptors
            ),
        },
        "operators": {
            "n_blocks": bundle.operators.n_blocks,
            "stored_block_entries": block_entries,
            "hamiltonian_dtype": bundle.operators.h_values.dtype.name,
            "overlap_dtype": bundle.operators.s_values.dtype.name,
            "has_nonzero_lattice_shifts": bool(
                np.any(bundle.operators.lattice_shift != 0)
            ),
            "hamiltonian_unit": "eV",
        },
        "filling": (
            None
            if filling is None
            else {
                "mode": filling.mode,
                "electron_count": filling.electron_count,
                "spin_degeneracy": filling.spin_degeneracy,
                "vbm_state_number": filling.vbm_state_number,
                "cbm_state_number": filling.cbm_state_number,
                "fermi_energy_ev": filling.fermi_energy_ev,
            }
        ),
        "backends": [
            {
                "name": item.name,
                "application": item.application,
                "precisions": list(item.precisions),
                "installed": item.installed,
                "dependency": item.dependency,
            }
            for item in backend_descriptors()
        ],
        "provenance": dict(bundle.provenance),
    }


def print_inspection(path: str | Path) -> None:
    print(json.dumps(inspect_bundle(path), indent=2, sort_keys=True))


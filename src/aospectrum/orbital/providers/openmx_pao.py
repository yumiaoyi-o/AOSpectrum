"""OpenMX PAO basis evaluation on CPU or one CUDA device."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from aospectrum.errors import FieldError
from aospectrum.model.basis import OrbitalBasisLayout
from aospectrum.model.structure import AtomicStructure
from aospectrum.orbital.model import GridSpec
from aospectrum.orbital.providers.base import FieldResources, OrbitalField
from aospectrum.orbital.providers.openmx_pao_data import OpenMXPAOData
from aospectrum.orbital.providers.openmx_radial import OpenMXRadialSpline


ANGSTROM_TO_BOHR = 1.889_726_125_457_828_1


@dataclass(frozen=True, slots=True)
class OpenMXPAOProvider:
    """Evaluate a basis whose coefficient descriptors use OpenMX ordering."""

    basis_identity: str
    pao_by_atomic_number: Mapping[int, OpenMXPAOData]

    def __post_init__(self) -> None:
        if not isinstance(self.basis_identity, str) or not self.basis_identity:
            raise FieldError("OpenMX provider basis_identity must not be empty")
        normalized: dict[int, OpenMXPAOData] = {}
        for atomic_number, pao in self.pao_by_atomic_number.items():
            if (
                isinstance(atomic_number, bool)
                or int(atomic_number) <= 0
                or not isinstance(pao, OpenMXPAOData)
            ):
                raise FieldError("OpenMX PAO mapping is invalid")
            normalized[int(atomic_number)] = pao
        if not normalized:
            raise FieldError("OpenMX provider requires at least one PAO")
        object.__setattr__(
            self,
            "pao_by_atomic_number",
            MappingProxyType(normalized),
        )

    def evaluate_orbital(
        self,
        structure: AtomicStructure,
        layout: OrbitalBasisLayout,
        coefficients: np.ndarray,
        grid: GridSpec,
        resources: FieldResources,
        *,
        precision: str,
    ) -> OrbitalField:
        if layout.basis_identity != self.basis_identity:
            raise FieldError("basis provider identity differs from AO layout")
        if layout.spinor_width != 1:
            raise FieldError("v1 OpenMX field provider does not render spinors")
        if not layout.has_real_space_descriptors:
            raise FieldError(
                "OpenMX field evaluation requires l/radial/harmonic descriptors"
            )
        expected = (
            np.dtype(np.complex64)
            if precision == "float32"
            else np.dtype(np.complex128)
        )
        values = np.asarray(coefficients)
        if values.shape != (layout.n_orbitals,) or values.dtype != expected:
            raise FieldError(
                "Orbital coefficients differ from layout or requested precision"
            )
        if not np.all(np.isfinite(values)):
            raise FieldError("Orbital coefficients must be finite")
        for atomic_number in np.unique(structure.atomic_numbers):
            if int(atomic_number) not in self.pao_by_atomic_number:
                raise FieldError(
                    f"OpenMX provider has no PAO for Z={int(atomic_number)}"
                )
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise FieldError(
                "OpenMX field evaluation requires the optional torch dependency"
            ) from exc
        try:
            device, device_label = _resolve_device(resources.device, torch)
            return self._evaluate_torch(
                structure,
                layout,
                values,
                grid.resolve_shape(structure.cell_angstrom),
                resources,
                precision,
                torch,
                device,
                device_label,
            )
        except FieldError:
            raise
        except RuntimeError as exc:
            raise FieldError(
                f"OpenMX field evaluation failed on {resources.device}: {exc}"
            ) from exc

    def _evaluate_torch(
        self,
        structure: AtomicStructure,
        layout: OrbitalBasisLayout,
        coefficients: np.ndarray,
        grid_shape: tuple[int, int, int],
        resources: FieldResources,
        precision: str,
        torch: Any,
        device: Any,
        device_label: str,
    ) -> OrbitalField:
        real_numpy = np.float32 if precision == "float32" else np.float64
        complex_numpy = np.complex64 if precision == "float32" else np.complex128
        real_torch = torch.float32 if precision == "float32" else torch.float64
        complex_torch = (
            torch.complex64 if precision == "float32" else torch.complex128
        )
        prepared = self._prepare_radials(
            structure,
            layout,
            real_numpy,
            real_torch,
            torch,
            device,
        )
        coefficient_tensor = torch.as_tensor(
            np.array(coefficients, dtype=complex_numpy, copy=True),
            dtype=complex_torch,
            device=device,
        )
        cell = np.asarray(structure.cell_angstrom, dtype=np.float64)
        positions = np.asarray(structure.positions_angstrom, dtype=np.float64)
        inverse_cell = np.linalg.inv(cell)
        atom_fractional = positions @ inverse_cell
        points = _half_open_grid_points(grid_shape, cell).reshape(-1, 3)
        output = np.empty(points.shape[0], dtype=complex_numpy)
        entries = _entries_by_atom(layout)
        angular = layout.angular_momentum
        radial = layout.radial_index
        harmonic = layout.real_harmonic_index
        assert angular is not None and radial is not None and harmonic is not None

        for start in range(0, points.shape[0], resources.chunk_points):
            stop = min(start + resources.chunk_points, points.shape[0])
            host_points = points[start:stop]
            point_tensor = torch.as_tensor(
                host_points,
                dtype=real_torch,
                device=device,
            )
            fractional = host_points @ inverse_cell
            accumulated = torch.zeros(
                stop - start,
                dtype=complex_torch,
                device=device,
            )
            for atom_index, coefficient_indices in entries.items():
                atomic_number = int(structure.atomic_numbers[atom_index])
                pao = self.pao_by_atomic_number[atomic_number]
                translations = _translation_indices(
                    fractional,
                    atom_fractional[atom_index],
                    cutoff_angstrom=pao.cutoff_bohr / ANGSTROM_TO_BOHR,
                    inverse_cell=inverse_cell,
                )
                for translation in translations:
                    image = positions[atom_index] + translation @ cell
                    image_tensor = torch.as_tensor(
                        image,
                        dtype=real_torch,
                        device=device,
                    )
                    relative = (point_tensor - image_tensor) * ANGSTROM_TO_BOHR
                    radius = torch.linalg.vector_norm(relative, dim=1)
                    tolerance = (
                        16.0
                        * torch.finfo(real_torch).eps
                        * max(1.0, pao.cutoff_bohr)
                    )
                    inside = radius <= pao.cutoff_bohr + tolerance
                    if not bool(torch.any(inside).item()):
                        continue
                    selected = torch.nonzero(inside, as_tuple=False).flatten()
                    selected_relative = relative[selected]
                    selected_radius = radius[selected]
                    harmonic_cache: dict[int, Any] = {}
                    radial_cache: dict[tuple[int, int], Any] = {}
                    for coefficient_index in coefficient_indices:
                        l_value = int(angular[coefficient_index])
                        radial_key = (
                            l_value,
                            int(radial[coefficient_index]),
                        )
                        if l_value not in harmonic_cache:
                            harmonic_cache[l_value] = _real_harmonics(
                                l_value,
                                selected_relative,
                                selected_radius,
                                real_torch,
                                torch,
                            )
                        if radial_key not in radial_cache:
                            radial_cache[radial_key] = prepared[
                                (atomic_number, *radial_key)
                            ].evaluate(selected_radius, torch=torch)
                        basis_values = radial_cache[radial_key] * harmonic_cache[
                            l_value
                        ][:, int(harmonic[coefficient_index])]
                        accumulated[selected] += (
                            coefficient_tensor[coefficient_index] * basis_values
                        )
            output[start:stop] = accumulated.detach().cpu().numpy()

        determinant = abs(float(np.linalg.det(cell)))
        voxel_volume_bohr3 = (
            determinant
            * ANGSTROM_TO_BOHR**3
            / float(math.prod(grid_shape))
        )
        return OrbitalField(
            values=output.reshape(grid_shape),
            grid_shape=grid_shape,
            cell_angstrom=cell,
            origin_angstrom=np.zeros(3, dtype=np.float64),
            voxel_volume_bohr3=voxel_volume_bohr3,
            compute_device=device_label,
            precision=precision,
        )

    def _prepare_radials(
        self,
        structure: AtomicStructure,
        layout: OrbitalBasisLayout,
        real_numpy: Any,
        real_torch: Any,
        torch: Any,
        device: Any,
    ) -> dict[tuple[int, int, int], Any]:
        angular = layout.angular_momentum
        radial = layout.radial_index
        assert angular is not None and radial is not None
        required = {
            (
                int(
                    structure.atomic_numbers[
                        int(layout.orbital_atom[index])
                    ]
                ),
                int(angular[index]),
                int(radial[index]),
            )
            for index in range(layout.n_orbitals)
        }
        prepared: dict[tuple[int, int, int], Any] = {}
        for atomic_number, l_value, radial_index in sorted(required):
            pao = self.pao_by_atomic_number[atomic_number]
            samples = np.asarray(
                pao.radial(l_value, radial_index),
                dtype=real_numpy,
            )
            mesh = np.asarray(pao.radial_mesh_bohr, dtype=real_numpy)
            prepared[(atomic_number, l_value, radial_index)] = (
                OpenMXRadialSpline.from_samples(
                    l_value,
                    mesh,
                    samples,
                ).to_torch(
                    torch=torch,
                    dtype=real_torch,
                    device=device,
                )
            )
        return prepared


def _resolve_device(requested: str, torch: Any) -> tuple[Any, str]:
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise FieldError(f"invalid field device {requested!r}") from exc
    if device.type == "cpu":
        if device.index is not None:
            raise FieldError("CPU field device must not include an index")
        return torch.device("cpu"), "cpu"
    if device.type != "cuda" or not torch.cuda.is_available():
        raise FieldError(f"CUDA field device is unavailable: {requested}")
    index = torch.cuda.current_device() if device.index is None else device.index
    if index < 0 or index >= torch.cuda.device_count():
        raise FieldError(f"CUDA field device index is invalid: {index}")
    torch.cuda.set_device(index)
    resolved = torch.device(f"cuda:{index}")
    return resolved, str(resolved)


def _half_open_grid_points(
    grid: tuple[int, int, int],
    cell: np.ndarray,
) -> np.ndarray:
    axes = tuple(
        np.arange(size, dtype=np.float64) / float(size) for size in grid
    )
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return fractional @ cell


def _entries_by_atom(layout: OrbitalBasisLayout) -> dict[int, tuple[int, ...]]:
    grouped: dict[int, list[int]] = {}
    for index, atom in enumerate(layout.orbital_atom):
        grouped.setdefault(int(atom), []).append(index)
    return {atom: tuple(indices) for atom, indices in grouped.items()}


def _translation_indices(
    chunk_fractional: np.ndarray,
    atom_fractional: np.ndarray,
    *,
    cutoff_angstrom: float,
    inverse_cell: np.ndarray,
) -> tuple[np.ndarray, ...]:
    delta = chunk_fractional - atom_fractional
    reciprocal_bounds = cutoff_angstrom * np.linalg.norm(inverse_cell, axis=0)
    tolerance = 32.0 * np.finfo(np.float64).eps * (
        1.0 + np.abs(delta).max(axis=0) + reciprocal_bounds
    )
    lower = np.ceil(delta.min(axis=0) - reciprocal_bounds - tolerance).astype(
        np.int64
    )
    upper = np.floor(delta.max(axis=0) + reciprocal_bounds + tolerance).astype(
        np.int64
    )
    if np.any(lower > upper):
        return ()
    return tuple(
        np.asarray(indices, dtype=np.float64)
        for indices in product(
            range(int(lower[0]), int(upper[0]) + 1),
            range(int(lower[1]), int(upper[1]) + 1),
            range(int(lower[2]), int(upper[2]) + 1),
        )
    )


def _comp2real_matrix(l_value: int) -> np.ndarray:
    dimension = 2 * l_value + 1
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    if l_value == 0:
        matrix[0, 0] = 1.0
        return matrix
    scale = 1.0 / math.sqrt(2.0)
    if l_value == 1:
        matrix[0, 0], matrix[0, 2] = scale, -scale
        matrix[1, 0], matrix[1, 2] = 1j * scale, 1j * scale
        matrix[2, 1] = 1.0
        return matrix
    if l_value == 2:
        matrix[0, 2] = 1.0
        matrix[1, 0], matrix[1, 4] = scale, scale
        matrix[2, 0], matrix[2, 4] = 1j * scale, -1j * scale
        matrix[3, 1], matrix[3, 3] = scale, -scale
        matrix[4, 1], matrix[4, 3] = 1j * scale, 1j * scale
        return matrix
    matrix[0, l_value] = 1.0
    odd = 0
    for row in range(1, dimension, 4):
        m_value = 2 * odd + 1
        matrix[row, l_value - m_value] = scale
        matrix[row, l_value + m_value] = -scale
        odd += 1
    even = 1
    for row in range(3, dimension, 4):
        m_value = 2 * even
        matrix[row, l_value - m_value] = scale
        matrix[row, l_value + m_value] = scale
        even += 1
    odd = 0
    for row in range(2, dimension, 4):
        m_value = 2 * odd + 1
        matrix[row, l_value - m_value] = 1j * scale
        matrix[row, l_value + m_value] = 1j * scale
        odd += 1
    even = 1
    for row in range(4, dimension, 4):
        m_value = 2 * even
        matrix[row, l_value - m_value] = 1j * scale
        matrix[row, l_value + m_value] = -1j * scale
        even += 1
    return matrix


def _associated_legendre(
    lmax: int,
    cosine: Any,
    real_dtype: Any,
    torch: Any,
) -> dict[tuple[int, int], Any]:
    sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
    result: dict[tuple[int, int], Any] = {
        (0, 0): torch.full(
            cosine.shape,
            1.0 / math.sqrt(4.0 * math.pi),
            dtype=real_dtype,
            device=cosine.device,
        )
    }
    for l_value in range(1, lmax + 1):
        result[(l_value, l_value)] = (
            -math.sqrt((2.0 * l_value + 1.0) / (2.0 * l_value))
            * sine
            * result[(l_value - 1, l_value - 1)]
        )
        result[(l_value, l_value - 1)] = (
            math.sqrt(2.0 * l_value + 1.0)
            * cosine
            * result[(l_value - 1, l_value - 1)]
        )
    for m_value in range(lmax + 1):
        for l_value in range(m_value + 2, lmax + 1):
            a = math.sqrt(
                (4.0 * l_value**2 - 1.0)
                / (l_value**2 - m_value**2)
            )
            b = math.sqrt(
                ((l_value - 1.0) ** 2 - m_value**2)
                / (4.0 * (l_value - 1.0) ** 2 - 1.0)
            )
            result[(l_value, m_value)] = a * (
                cosine * result[(l_value - 1, m_value)]
                - b * result[(l_value - 2, m_value)]
            )
    return result


def _real_harmonics(
    l_value: int,
    relative: Any,
    radius: Any,
    real_dtype: Any,
    torch: Any,
) -> Any:
    if l_value == 0:
        return torch.full(
            (relative.shape[0], 1),
            1.0 / math.sqrt(4.0 * math.pi),
            dtype=real_dtype,
            device=relative.device,
        )
    nonzero = radius > 16.0 * torch.finfo(real_dtype).eps
    safe = torch.where(nonzero, radius, torch.ones_like(radius))
    cosine = torch.clamp(relative[:, 2] / safe, min=-1.0, max=1.0)
    phi = torch.remainder(
        torch.atan2(relative[:, 1], relative[:, 0]),
        2.0 * math.pi,
    )
    plm = _associated_legendre(l_value, cosine, real_dtype, torch)
    complex_dtype = (
        torch.complex64 if real_dtype == torch.float32 else torch.complex128
    )
    exp_one = torch.exp(1j * phi.to(complex_dtype))
    exponentials = {
        0: torch.ones_like(exp_one),
        1: exp_one,
    }
    for order in range(2, l_value + 1):
        exponentials[order] = exponentials[order - 1] * exp_one
    harmonics: dict[int, Any] = {}
    for order in range(l_value + 1):
        positive = plm[(l_value, order)].to(complex_dtype) * exponentials[order]
        harmonics[order] = positive
        if order:
            harmonics[-order] = ((-1) ** order) * positive.conj()
    complex_values = torch.stack(
        [harmonics[order] for order in range(-l_value, l_value + 1)],
        dim=1,
    )
    transform = torch.as_tensor(
        _comp2real_matrix(l_value),
        dtype=complex_dtype,
        device=relative.device,
    )
    real_values = (complex_values @ transform.conj().T).real
    return torch.where(nonzero[:, None], real_values, torch.zeros_like(real_values))

"""OpenMX cubic interpolation for PAO radial functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TorchRadialSpline:
    mesh_bohr: Any
    interval_coefficients: Any
    origin_coefficients: Any

    def evaluate(self, radius_bohr: Any, *, torch: Any) -> Any:
        radius = radius_bohr.reshape(-1)
        result = torch.zeros_like(radius)
        below = radius < self.mesh_bohr[0]
        if bool(torch.any(below).item()):
            query = radius[below]
            coefficients = self.origin_coefficients
            result[below] = (
                (
                    coefficients[3] * query + coefficients[2]
                )
                * query
                + coefficients[1]
            ) * query + coefficients[0]
        inside = (radius >= self.mesh_bohr[0]) & (
            radius <= self.mesh_bohr[-1]
        )
        if bool(torch.any(inside).item()):
            query = radius[inside]
            right = torch.searchsorted(self.mesh_bohr, query)
            right = torch.clamp(right, min=1, max=self.mesh_bohr.numel() - 1)
            left = right - 1
            scaled = (query - self.mesh_bohr[left]) / (
                self.mesh_bohr[right] - self.mesh_bohr[left]
            )
            coefficients = self.interval_coefficients[left]
            result[inside] = (
                (
                    coefficients[:, 3] * scaled + coefficients[:, 2]
                )
                * scaled
                + coefficients[:, 1]
            ) * scaled + coefficients[:, 0]
        return result.reshape(radius_bohr.shape)


@dataclass(frozen=True, slots=True)
class OpenMXRadialSpline:
    angular_momentum: int
    mesh_bohr: np.ndarray
    interval_coefficients: np.ndarray
    origin_coefficients: np.ndarray

    @classmethod
    def from_samples(
        cls,
        angular_momentum: int,
        mesh_bohr: np.ndarray,
        values: np.ndarray,
    ) -> "OpenMXRadialSpline":
        mesh = np.asarray(mesh_bohr)
        samples = np.asarray(values)
        dtype = (
            np.float32
            if mesh.dtype == np.float32 and samples.dtype == np.float32
            else np.float64
        )
        mesh = np.asarray(mesh, dtype=dtype)
        samples = np.asarray(samples, dtype=dtype)
        if (
            angular_momentum < 0
            or mesh.ndim != 1
            or samples.shape != mesh.shape
            or mesh.size < 6
            or not np.all(np.isfinite(mesh))
            or not np.all(np.isfinite(samples))
            or np.any(mesh < 0.0)
            or np.any(np.diff(mesh) <= 0.0)
        ):
            raise ValueError("invalid OpenMX radial samples")
        intervals = _interval_coefficients(mesh, samples)
        origin = _origin_coefficients(
            angular_momentum,
            mesh,
            samples,
            intervals,
        )
        for array in (mesh, intervals, origin):
            array.setflags(write=False)
        return cls(angular_momentum, mesh, intervals, origin)

    def to_torch(self, *, torch: Any, dtype: Any, device: Any) -> TorchRadialSpline:
        return TorchRadialSpline(
            mesh_bohr=torch.as_tensor(
                np.array(self.mesh_bohr, copy=True),
                dtype=dtype,
                device=device,
            ),
            interval_coefficients=torch.as_tensor(
                np.array(self.interval_coefficients, copy=True),
                dtype=dtype,
                device=device,
            ),
            origin_coefficients=torch.as_tensor(
                np.array(self.origin_coefficients, copy=True),
                dtype=dtype,
                device=device,
            ),
        )


def _interval_coefficients(
    mesh: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    coefficients = np.empty((mesh.size - 1, 4), dtype=mesh.dtype)
    for left in range(mesh.size - 1):
        h2 = mesh[left + 1] - mesh[left]
        f2, f3 = values[left], values[left + 1]
        if left == 0:
            h3 = mesh[2] - mesh[1]
            h1 = -(h2 + h3)
            f1 = f4 = values[2]
        elif left == mesh.size - 2:
            h1 = mesh[left] - mesh[left - 1]
            h3 = -(h1 + h2)
            f1 = f4 = values[left - 1]
        else:
            h1 = mesh[left] - mesh[left - 1]
            h3 = mesh[left + 2] - mesh[left + 1]
            f1, f4 = values[left - 1], values[left + 2]
        g1 = ((f3 - f2) * h1 / h2 + (f2 - f1) * h2 / h1) / (h1 + h2)
        g2 = ((f4 - f3) * h2 / h3 + (f3 - f2) * h3 / h2) / (h2 + h3)
        scaled_g1, scaled_g2 = h2 * g1, h2 * g2
        coefficients[left] = (
            f2,
            scaled_g1,
            -3.0 * f2 - 2.0 * scaled_g1 + 3.0 * f3 - scaled_g2,
            2.0 * f2 + scaled_g1 - 2.0 * f3 + scaled_g2,
        )
    return coefficients


def _origin_coefficients(
    angular_momentum: int,
    mesh: np.ndarray,
    values: np.ndarray,
    intervals: np.ndarray,
) -> np.ndarray:
    reference = 4
    left = reference - 1
    coefficients = intervals[left]
    radius = mesh[reference]
    value = values[reference]
    derivative = (
        coefficients[1] + 2.0 * coefficients[2] + 3.0 * coefficients[3]
    ) / (mesh[reference] - mesh[left])
    origin = np.zeros(4, dtype=mesh.dtype)
    if angular_momentum == 0:
        origin[2] = 0.5 * derivative / radius
        origin[0] = value - origin[2] * radius**2
    elif angular_momentum == 1:
        origin[3] = (radius * derivative - value) / (2.0 * radius**3)
        origin[1] = derivative - 3.0 * origin[3] * radius**2
    else:
        origin[2] = (3.0 * value - radius * derivative) / radius**2
        origin[3] = (value - origin[2] * radius**2) / radius**3
    return origin

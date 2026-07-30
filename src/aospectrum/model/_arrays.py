"""Small helpers for immutable NumPy-backed domain values."""

from __future__ import annotations

from typing import Any

import numpy as np

from aospectrum.errors import ContractError


def frozen_array(
    value: Any,
    *,
    name: str,
    ndim: int,
    dtype: np.dtype[Any] | str | None = None,
    kinds: str | None = None,
) -> np.ndarray:
    """Return a finite, C-contiguous, read-only array.

    Mutable caller arrays are copied. Read-only C-contiguous arrays, including
    arrays opened with ``numpy.load(..., mmap_mode="r")``, retain their storage.
    """

    array = np.asarray(value, dtype=dtype)
    if array.ndim != ndim:
        raise ContractError(
            f"{name} must have {ndim} dimensions, got shape {array.shape}"
        )
    if kinds is not None and array.dtype.kind not in kinds:
        raise ContractError(f"{name} has unsupported dtype {array.dtype}")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ContractError(f"{name} must contain only finite values")
    if not array.flags.c_contiguous or array.flags.writeable:
        array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


def scalar_family_bits(dtype: np.dtype[Any] | str) -> int:
    scalar = np.dtype(dtype)
    if scalar in (np.dtype(np.float32), np.dtype(np.complex64)):
        return 32
    if scalar in (np.dtype(np.float64), np.dtype(np.complex128)):
        return 64
    raise ContractError(
        "operator values must use float32, complex64, float64, or complex128"
    )


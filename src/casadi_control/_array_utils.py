"""Shared numeric array coercion helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def as_optional_1d_float_array(x: Any) -> Optional[np.ndarray]:
    """Convert scalar-like input to a one-dimensional float array."""
    if x is None:
        return None
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array or scalar, got shape {arr.shape}")
    return arr


def as_positive_optional_1d_float_array(
    x: Any,
    *,
    name: str,
    floor: float,
) -> Optional[np.ndarray]:
    """Convert input to a positive 1D float array with optional lower clipping."""
    arr = as_optional_1d_float_array(x)
    if arr is None:
        return None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive; got min={arr.min()}")
    if floor is not None and floor > 0:
        arr = np.maximum(arr, float(floor))
    return arr


def as_sized_1d_float_vector(
    x: Any,
    size: int,
    *,
    name: str,
    broadcast_scalar: bool = False,
) -> np.ndarray:
    """Convert input to a dense vector of a fixed length."""
    arr = np.asarray(x, float).reshape(-1)
    if arr.size == 1 and broadcast_scalar and size != 1:
        return np.full((size,), float(arr[0]))
    if arr.size != size:
        raise ValueError(f"{name} has size {arr.size}, expected {size}")
    return arr

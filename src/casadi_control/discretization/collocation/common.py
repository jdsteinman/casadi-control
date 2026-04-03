"""Shared internal helpers for collocation modules."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..._array_utils import as_optional_1d_float_array
from .decode import CollocationMeta


def as_1d_float_array(x: Any) -> np.ndarray:
    """Convert input to a one-dimensional float array."""
    arr = as_optional_1d_float_array(x)
    if arr is None:
        raise ValueError("Expected array-like input, got None")
    return arr


def validate_s_mesh(s_mesh: np.ndarray, *, expected_size: int | None = None) -> np.ndarray:
    """Validate normalized mesh monotonicity and endpoint convention."""
    s = np.asarray(s_mesh, float).reshape(-1)
    if expected_size is not None and s.size != expected_size:
        raise ValueError(f"s_mesh must have length {expected_size}, got {s.size}")
    if s.size < 2:
        raise ValueError("s_mesh must have length >= 2")
    if not np.all(np.diff(s) > 0.0):
        raise ValueError("s_mesh must be strictly increasing")
    if abs(float(s[0]) - 0.0) > 1e-12 or abs(float(s[-1]) - 1.0) > 1e-12:
        raise ValueError("s_mesh must start at 0 and end at 1 (within tolerance)")
    return s


def normalize_time_grid_to_s_mesh(t_mesh: np.ndarray, *, t0: float) -> np.ndarray:
    """Convert a physical time grid to a normalized mesh in ``[0, 1]``."""
    t = as_1d_float_array(t_mesh)
    if t.size < 2:
        raise ValueError("time grid must have length >= 2")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("time grid must be strictly increasing")

    denom = float(t[-1] - t0)
    if denom <= 0.0:
        raise ValueError("time grid must satisfy t_mesh[-1] > t0")

    s = (t - float(t0)) / denom
    s[0] = 0.0
    s[-1] = 1.0
    return validate_s_mesh(s)


def is_scaled_meta(meta: Dict[str, Any] | CollocationMeta) -> bool:
    """Return whether collocation metadata indicates scaled solver coordinates."""
    if isinstance(meta, CollocationMeta):
        return meta.space.lower() == "scaled"
    return str(meta.get("space", "physical")).lower() == "scaled"

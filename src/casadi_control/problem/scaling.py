"""Scaling metadata and coordinate transforms for OCP models.

This module defines :class:`casadi_control.problem.scaling.Scaling`, which stores
variable/time reference scales and provides helpers for mapping between physical
and scaled OCP coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Literal

import numpy as np


def _as_1d_array(x):
    """Convert scalar-like input to a one-dimensional float array."""
    if x is None:
        return None
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array or scalar, got shape {arr.shape}")
    return arr

def _as_pos_1d_array(x, *, name: str, floor: float) -> Optional[np.ndarray]:
    """Convert input to positive 1D float array with optional lower clipping."""
    arr = _as_1d_array(x)
    if arr is None:
        return None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive; got min={arr.min()}")
    if floor is not None and floor > 0:
        arr = np.maximum(arr, float(floor))
    return arr


@dataclass(frozen=True)
class Scaling:
    """Immutable scaling specification for OCP coordinates.

    Parameters
    ----------
    space : {"physical", "scaled"}, optional
        Coordinate space represented by the associated OCP callbacks and bounds.
        Use ``"scaled"`` when the OCP is already expressed in normalized
        coordinates.
    x_ref, u_ref, p_ref : array-like, optional
        Positive diagonal reference vectors for state, control, and parameter
        scaling.
    t_ref : float, optional
        Positive time reference used for dimensionless time normalization.
    t0_phys : float, optional
        Physical initial time used when reconstructing physical time from
        scaled-time coordinates.
    J_ref : float, optional
        Positive objective normalization factor.
    defect_ref, path_ref, state_ref, bnd_ref : array-like, optional
        Positive scaling vectors for defect, path, state, and boundary
        constraints, interpreted in the OCP's current coordinate space.
    floor : float, optional
        Lower bound applied to positive scaling vectors for numerical safety.


    Methods
    -------
    scale_x
    unscale_x
    scale_u
    unscale_u
    scale_p
    unscale_p
    t_phys_from_t
    tf_phys_from_tf
    scale_bounds
    unscale_bounds
    defect_scale_vec
    path_scale_vec
    state_scale_vec
    bnd_scale_vec


    Notes
    -----
    Variable scaling uses diagonal maps:

    - ``x_phys = diag(x_ref) x_hat``
    - ``u_phys = diag(u_ref) u_hat``
    - ``p_phys = diag(p_ref) p_hat``

    If ``t_ref`` is provided, physical time is reconstructed as
    ``t_phys = t0_phys + t_ref * (t_ocp - t0_ocp)`` in scaled space.
    """
    space: Literal["physical", "scaled"] = "physical"

    # variable refs (in physical units; used for reconstruction even in scaled space)
    x_ref: Optional[np.ndarray] = None   # (nx,)
    u_ref: Optional[np.ndarray] = None   # (nu,)
    p_ref: Optional[np.ndarray] = None   # (np,)

    # time normalization
    t_ref: Optional[float] = None        # scalar > 0
    t0_phys: Optional[float] = None      # scalar (required if t_ref is used and space=="scaled")

    # objective normalization (optional)
    J_ref: Optional[float] = None

    # optional constraint scaling (in current OCP coordinates)
    defect_ref: Optional[np.ndarray] = None  # (nx,)
    path_ref: Optional[np.ndarray] = None    # (m_path,)
    state_ref: Optional[np.ndarray] = None   # (m_state,)
    bnd_ref: Optional[np.ndarray] = None     # (m_bnd,)

    floor: float = 1e-12

    def __post_init__(self):
        # variable refs: must be positive if provided
        object.__setattr__(self, "x_ref", _as_pos_1d_array(self.x_ref, name="x_ref", floor=self.floor))
        object.__setattr__(self, "u_ref", _as_pos_1d_array(self.u_ref, name="u_ref", floor=self.floor))
        object.__setattr__(self, "p_ref", _as_pos_1d_array(self.p_ref, name="p_ref", floor=self.floor))

        # time
        if self.t_ref is not None:
            t_ref = float(self.t_ref)
            if not np.isfinite(t_ref) or t_ref <= 0:
                raise ValueError("t_ref must be finite and positive")
            object.__setattr__(self, "t_ref", t_ref)

        if self.t0_phys is not None:
            t0p = float(self.t0_phys)
            if not np.isfinite(t0p):
                raise ValueError("t0_phys must be finite")
            object.__setattr__(self, "t0_phys", t0p)

        if self.J_ref is not None:
            J = float(self.J_ref)
            if not np.isfinite(J) or J <= 0:
                raise ValueError("J_ref must be finite and positive")
            object.__setattr__(self, "J_ref", J)

        # constraint refs (allow None; if provided must be positive)
        object.__setattr__(self, "defect_ref", _as_pos_1d_array(self.defect_ref, name="defect_ref", floor=self.floor))
        object.__setattr__(self, "path_ref", _as_pos_1d_array(self.path_ref, name="path_ref", floor=self.floor))
        object.__setattr__(self, "state_ref", _as_pos_1d_array(self.state_ref, name="state_ref", floor=self.floor))
        object.__setattr__(self, "bnd_ref", _as_pos_1d_array(self.bnd_ref, name="bnd_ref", floor=self.floor))

        if self.space not in ("physical", "scaled"):
            raise ValueError("space must be 'physical' or 'scaled'")

        # if time normalized + scaled space, we need t0_phys for reconstruction
        if self.space == "scaled" and self.t_ref is not None and self.t0_phys is None:
            raise ValueError("Scaling with t_ref in scaled space requires t0_phys for reconstruction")

    # --------------------------
    # Variable scaling operators
    # --------------------------
    def _x_ref(self, nx: int) -> np.ndarray:
        return np.ones(nx) if self.x_ref is None else self.x_ref

    def _u_ref(self, nu: int) -> np.ndarray:
        return np.ones(nu) if self.u_ref is None else self.u_ref

    def _p_ref(self, n_p: int) -> np.ndarray:
        return np.ones(n_p) if n_p == 0 or self.p_ref is None else self.p_ref

    def scale_x(self, x_phys: np.ndarray, *, nx: int) -> np.ndarray:
        """Scale physical state coordinates to solver coordinates."""
        return np.asarray(x_phys, float).reshape(-1) / self._x_ref(nx)

    def unscale_x(self, x_hat: np.ndarray, *, nx: int) -> np.ndarray:
        """Map scaled state coordinates back to physical coordinates."""
        return np.asarray(x_hat, float).reshape(-1) * self._x_ref(nx)

    def scale_u(self, u_phys: np.ndarray, *, nu: int) -> np.ndarray:
        """Scale physical control coordinates to solver coordinates."""
        return np.asarray(u_phys, float).reshape(-1) / self._u_ref(nu)

    def unscale_u(self, u_hat: np.ndarray, *, nu: int) -> np.ndarray:
        """Map scaled control coordinates back to physical coordinates."""
        return np.asarray(u_hat, float).reshape(-1) * self._u_ref(nu)

    def scale_p(self, p_phys: np.ndarray, *, n_p: int) -> np.ndarray:
        """Scale physical parameter coordinates to solver coordinates."""
        if n_p == 0:
            return np.asarray([], float)
        return np.asarray(p_phys, float).reshape(-1) / self._p_ref(n_p)

    def unscale_p(self, p_hat: np.ndarray, *, n_p: int) -> np.ndarray:
        """Map scaled parameter coordinates back to physical coordinates."""
        if n_p == 0:
            return np.asarray([], float)
        return np.asarray(p_hat, float).reshape(-1) * self._p_ref(n_p)

    # --------------------------
    # Time scaling operators
    # --------------------------
    def t_phys_from_t(self, t_ocp: Any, *, t0_ocp: float) -> Any:
        """
        Map OCP time coordinate to physical time.

        - If space=="physical": identity
        - If space=="scaled" and t_ref set: t_phys = t0_phys + t_ref*(t_ocp - t0_ocp)
          (your scaled OCP uses t0=0 usually; this handles general offsets)
        """
        if self.t_ref is None or self.space == "physical":
            return t_ocp
        assert self.t0_phys is not None
        return float(self.t0_phys) + float(self.t_ref) * (t_ocp - float(t0_ocp))

    def tf_phys_from_tf(self, tf_ocp: Any) -> Any:
        """Map OCP final-time coordinate to physical duration."""
        if self.t_ref is None or self.space == "physical":
            return tf_ocp
        return float(self.t_ref) * tf_ocp

    # --------------------------
    # Bounds helpers
    # --------------------------
    def scale_bounds(self, bounds: Optional[Tuple[np.ndarray, np.ndarray]], *, ref: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Scale lower/upper bounds component-wise by ``ref``."""
        if bounds is None:
            return None
        lb, ub = bounds
        lb = np.asarray(lb, float).reshape(-1) / ref
        ub = np.asarray(ub, float).reshape(-1) / ref
        return lb, ub

    def unscale_bounds(self, bounds: Optional[Tuple[np.ndarray, np.ndarray]], *, ref: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Map scaled lower/upper bounds to physical bounds using ``ref``."""
        if bounds is None:
            return None
        lb, ub = bounds
        lb = np.asarray(lb, float).reshape(-1) * ref
        ub = np.asarray(ub, float).reshape(-1) * ref
        return lb, ub

    # --------------------------
    # Constraint scaling vectors
    # --------------------------
    def defect_scale_vec(self, *, nx: int) -> np.ndarray:
        """
        Returns a vector s.t. you should scale defect residuals as: r_scaled = r / vec.
        Default:
        - physical space: x_ref
        - scaled space: ones (avoid double-scaling)
        """
        if self.defect_ref is not None:
            return self.defect_ref
        if self.space == "scaled":
            return np.ones(nx)
        return self._x_ref(nx)

    def path_scale_vec(self, *, m: int) -> np.ndarray:
        if self.path_ref is not None:
            if self.path_ref.size != m:
                raise ValueError(f"path_ref has size {self.path_ref.size}, expected {m}")
            return self.path_ref
        return np.ones(m)

    def state_scale_vec(self, *, m: int) -> np.ndarray:
        if self.state_ref is not None:
            if self.state_ref.size != m:
                raise ValueError(f"state_ref has size {self.state_ref.size}, expected {m}")
            return self.state_ref
        return np.ones(m)

    def bnd_scale_vec(self, *, m: int) -> np.ndarray:
        if self.bnd_ref is not None:
            if self.bnd_ref.size != m:
                raise ValueError(f"bnd_ref has size {self.bnd_ref.size}, expected {m}")
            return self.bnd_ref
        return np.ones(m)


__all__ = ["Scaling"]

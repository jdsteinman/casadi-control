"""Optimal control problem specification.

This module defines the immutable :class:`casadi_control.problem.ocp.OCP`
dataclass used to describe an optimal control problem (OCP).  The class
stores the system dynamics, objective terms, constraints, bounds, and
optional scaling metadata as callable objects.

The problems represented by :class:`OCP` have the general form

.. math::

    \\begin{aligned}
    \\min_{x,\\,u,\\,p}\\quad
    & \\phi\\big(x(t_0),x(t_f),p\\big)
      + \\int_{t_0}^{t_f} \\ell\\big(x(t),u(t),p,t\\big)\\,dt
    \\\\[6pt]
    \\text{s.t.}\\quad
    & x'(t) = f\\big(x(t),u(t),p,t\\big)
    & \\forall t \\in (t_0,t_f)
    \\\\
    &
    c\\big(x(t),u(t),p,t\\big) \\le 0
    & \\forall t \\in [t_0,t_f]
    \\\\
    &
    s\\big(x(t),p,t\\big) \\le 0
    & \\forall t \\in [t_0,t_f]
    \\\\
    &
    b\\big(x(t_0),x(t_f)\\big) = 0
    \\\\
    &
    u(t) \\in \\mathcal{U}
    & \\forall t \\in [t_0,t_f].
    \\end{aligned}

Here

* :math:`x(t) \\in \\mathbb{R}^{n_x}` is the state,
* :math:`u(t) \\in \\mathbb{R}^{n_u}` is the control,
* :math:`p \\in \\mathbb{R}^{n_p}` is a vector of parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Tuple, Union, Literal

import numpy as np
import casadi as ca

from .scaling import Scaling


TfSpec = Union[float, Tuple[float, float]]  # fixed tf or (lb, ub) for free tf


def _as_1d_array(x):
    """Convert scalar-like input to a one-dimensional float array.

    Parameters
    ----------
    x : Any
        Scalar or array-like input, or ``None``.

    Returns
    -------
    ndarray or None
        One-dimensional ``float`` array, or ``None`` if ``x`` is ``None``.
    """
    if x is None:
        return None
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array or scalar, got shape {arr.shape}")
    return arr



@dataclass(frozen=True)
class OCP:
    """Immutable optimal control problem model.

    Parameters
    ----------
    n_x, n_u, n_p : int
        Dimensions of state, control, and parameter vectors.
    t0 : float, optional
        Initial time in the OCP's time coordinate.
    tf : float or tuple[float, float], optional
        Final time, either fixed or bounded for free-final-time problems.
    f_dyn : callable
        Dynamics callback with signature
        ``f_dyn(x, u, p, t) -> (n_x, 1)``.
    l_run : callable, optional
        Running-cost callback with signature
        ``l_run(x, u, p, t) -> scalar``.
    l_end : callable, optional
        Endpoint-cost callback with signature
        ``l_end(x0, xf, p, t0, tf) -> scalar``.
    bnd_constr : callable, optional
        Boundary-constraint callback with signature
        ``bnd_constr(x0, xf, p, t0, tf) -> (m_b, 1)``.
    path_constr : callable, optional
        Path-constraint callback with signature
        ``path_constr(x, u, p, t) -> (m_c, 1)``.
    state_constr : callable, optional
        State-only path-constraint callback with signature
        ``state_constr(x, p, t) -> (m_s, 1)``.
    x_bounds, u_bounds, p_bounds : tuple[array, array], optional
        Lower and upper bounds for decision components.
    x0_fixed, p0_guess : array-like, optional
        Fixed initial state and nominal parameter vector.
    scaling : Scaling, optional
        Scaling metadata attached to this OCP coordinate system.

    Methods
    -------
    validate
    scaled

    Notes
    -----
    All constraint callbacks should return CasADi-compatible column vectors
    ``(m, 1)`` or empty vectors when not active.
    """
    n_x: int
    n_u: int
    n_p: int = 0

    t0: float = 0.0
    tf: TfSpec = 1.0

    # Dynamics and costs
    f_dyn: Optional[Callable] = None        # x' = f(x,u,p,t)
    l_run: Optional[Callable] = None        # running cost (integrand over physical time)
    l_end: Optional[Callable] = None        # endpoint cost

    # Constraints
    bnd_constr: Optional[Callable] = None   # b(x0,xf,p,t0,tf) = 0
    path_constr: Optional[Callable] = None  # c(x,u,p,t) <= 0
    state_constr: Optional[Callable] = None # s(x,p,t) <= 0

    # Bounds
    x_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    u_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    p_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # Initial / nominal values
    x0_fixed: Optional[np.ndarray] = None
    p0_guess: Optional[np.ndarray] = None

    # Scaling info (present iff this OCP is in scaled coordinates)
    scaling: Optional[Scaling] = None

    def __post_init__(self):
        object.__setattr__(self, "x0_fixed", _as_1d_array(self.x0_fixed))
        object.__setattr__(self, "p0_guess", _as_1d_array(self.p0_guess))

        if self.x0_fixed is not None and self.x0_fixed.shape[0] != self.n_x:
            raise ValueError("x0_fixed has wrong dimension")

        if self.n_p == 0:
            if self.p_bounds is not None or self.p0_guess is not None:
                raise ValueError("n_p=0 but parameter data was provided")
        else:
            if self.p0_guess is not None and self.p0_guess.shape[0] != self.n_p:
                raise ValueError("p0_guess has wrong dimension")

        if isinstance(self.tf, tuple):
            lb, ub = self.tf
            if not (lb < ub):
                raise ValueError("tf bounds must satisfy lb < ub")

    def validate(self) -> None:
        """Validate callback signatures and output shapes.

        Raises
        ------
        ValueError
            If required callbacks are missing, or if callback outputs do not
            match expected scalar/vector conventions.
        """
        if self.f_dyn is None:
            raise ValueError("OCP.f_dyn must be provided")
        if self.l_run is None and self.l_end is None:
            raise ValueError("At least one of l_run or l_end must be provided")

        def _is_numeric_scalar(z: Any) -> bool:
            return isinstance(z, (int, float, np.integer, np.floating))

        def _shape(z: Any):
            return getattr(z, "shape", None)

        def _is_scalar_like(z: Any) -> bool:
            if _is_numeric_scalar(z):
                return True
            return _shape(z) in [(1, 1), (1,), ()]

        def _check_column(expr: Any, name: str) -> None:
            shp = _shape(expr)
            if shp is None:
                raise ValueError(f"{name} must return a CasADi column vector (m,1) or empty; got type {type(expr)}")
            if shp in [(0, 0), (0, 1), (0,)]:
                return
            if len(shp) != 2 or shp[1] != 1:
                raise ValueError(f"{name} must be a column vector (m,1) or empty; got shape {shp}")

        x = ca.MX.sym("x", self.n_x)  # type: ignore[arg-type]
        u = ca.MX.sym("u", self.n_u)  # type: ignore[arg-type]
        p = ca.MX.sym("p", self.n_p)  # type: ignore[arg-type]
        t = ca.MX.sym("t")            # type: ignore[arg-type]

        fx = self.f_dyn(x, u, p, t)
        if fx.shape != (self.n_x, 1):
            raise ValueError(f"f_dyn must return shape (nx,1); got {fx.shape}")

        if self.l_run is not None:
            lx = self.l_run(x, u, p, t)
            if not _is_scalar_like(lx):
                raise ValueError(f"l_run must be scalar-like; got shape={_shape(lx)} type={type(lx)}")

        x0 = ca.MX.sym("x0", self.n_x)  # type: ignore[arg-type]
        xf = ca.MX.sym("xf", self.n_x)  # type: ignore[arg-type]
        t0 = ca.MX.sym("t0")            # type: ignore[arg-type]
        tf = ca.MX.sym("tf")            # type: ignore[arg-type]

        if self.l_end is not None:
            phix = self.l_end(x0, xf, p, t0, tf)
            if not _is_scalar_like(phix):
                raise ValueError(f"l_end must be scalar-like; got shape={_shape(phix)} type={type(phix)}")

        if self.bnd_constr is not None:
            _check_column(self.bnd_constr(x0, xf, p, t0, tf), "bnd_constr")
        if self.path_constr is not None:
            _check_column(self.path_constr(x, u, p, t), "path_constr")
        if self.state_constr is not None:
            _check_column(self.state_constr(x, p, t), "state_constr")

    def scaled(self, scaling: Scaling) -> "OCP":
        """
        Return a coordinate-transformed OCP in scaled variables.

        Parameters
        ----------
        scaling : Scaling
            Variable/time scaling specification.

        Returns
        -------
        OCP
            New OCP instance whose callbacks operate in scaled coordinates.

        Convention:
        - The returned OCP lives in "scaled" space (scaling.space == "scaled").
        - User functions (original f_dyn, costs, constraints) are evaluated in physical units.
        - The transcription operates in the returned OCP's time coordinate:
            * if scaling.t_ref is set: dimensionless time with t0_hat typically 0
            * else: physical time
        """
        # Build refs
        x_ref = np.ones(self.n_x) if scaling.x_ref is None else scaling.x_ref
        u_ref = np.ones(self.n_u) if scaling.u_ref is None else scaling.u_ref
        p_ref = np.ones(self.n_p) if self.n_p == 0 or scaling.p_ref is None else scaling.p_ref
        t_ref = scaling.t_ref

        if x_ref.shape[0] != self.n_x:
            raise ValueError("scaling.x_ref has wrong dimension")
        if u_ref.shape[0] != self.n_u:
            raise ValueError("scaling.u_ref has wrong dimension")
        if self.n_p > 0 and p_ref.shape[0] != self.n_p:
            raise ValueError("scaling.p_ref has wrong dimension")

        # Ensure we can reconstruct physical time if time-normalizing
        if t_ref is not None and scaling.t0_phys is None:
            scaling = replace(scaling, t0_phys=float(self.t0))

        # Returned scaling is explicitly "scaled"
        scaling_hat = replace(scaling, space="scaled")

        x_ref_dm = ca.DM(x_ref)
        u_ref_dm = ca.DM(u_ref)
        p_ref_dm = ca.DM(p_ref) if self.n_p > 0 else ca.DM([])

        t0_phys = float(self.t0)

        # Scaled time starts at 0 if normalized; otherwise preserve physical t0
        t0_hat = 0.0 if t_ref is not None else float(self.t0)

        def _t_phys(t_hat_or_t_phys):
            if t_ref is None:
                return t_hat_or_t_phys
            return t0_phys + float(t_ref) * t_hat_or_t_phys

        def phys_from_scaled(xh, uh, ph, t_hat_or_t_phys):
            x = ca.diag(x_ref_dm) @ xh
            u = ca.diag(u_ref_dm) @ uh
            p = (ca.diag(p_ref_dm) @ ph) if self.n_p > 0 else ph
            t = _t_phys(t_hat_or_t_phys)
            return x, u, p, t

        def f_dyn_hat(xh, uh, ph, t_hat_or_t_phys):
            x, u, p, t = phys_from_scaled(xh, uh, ph, t_hat_or_t_phys)
            assert self.f_dyn is not None
            fx = self.f_dyn(x, u, p, t)

            # xh' = (dt_phys/dt_hat) * (1/x_ref) * f
            if t_ref is None:
                return ca.diag(1.0 / x_ref_dm) @ fx
            return ca.diag(float(t_ref) / x_ref_dm) @ fx

        l_run_hat: Optional[Callable] = None
        if self.l_run is not None:
            def _l_run_hat(xh, uh, ph, t_hat_or_t_phys):
                x, u, p, t = phys_from_scaled(xh, uh, ph, t_hat_or_t_phys)
                assert self.l_run is not None
                val = self.l_run(x, u, p, t)
                # Integral in transcription is over t_hat -> convert dt_phys = t_ref dt_hat
                if t_ref is None:
                    return val
                return float(t_ref) * val
            l_run_hat = _l_run_hat

        l_end_hat: Optional[Callable] = None
        if self.l_end is not None:
            def _l_end_hat(x0h, xfh, ph, t0_hat_in, tf_hat_in):
                x0 = ca.diag(x_ref_dm) @ x0h
                xf = ca.diag(x_ref_dm) @ xfh
                p  = (ca.diag(p_ref_dm) @ ph) if self.n_p > 0 else ph

                assert self.l_end is not None
                if t_ref is None:
                    t0p = t0_hat_in
                    tfp = tf_hat_in
                else:
                    t0p = t0_phys
                    tfp = t0_phys + float(t_ref) * tf_hat_in
                return self.l_end(x0, xf, p, t0p, tfp)
            l_end_hat = _l_end_hat

        bnd_constr_hat: Optional[Callable[..., Any]] = None
        if self.bnd_constr is not None:
            def _bnd_constr_hat(x0h, xfh, ph, t0_hat_in, tf_hat_in):
                x0 = ca.diag(x_ref_dm) @ x0h
                xf = ca.diag(x_ref_dm) @ xfh
                p  = (ca.diag(p_ref_dm) @ ph) if self.n_p > 0 else ph

                assert self.bnd_constr is not None
                if t_ref is None:
                    t0p = t0_hat_in
                    tfp = tf_hat_in
                else:
                    t0p = t0_phys
                    tfp = t0_phys + float(t_ref) * tf_hat_in
                return self.bnd_constr(x0, xf, p, t0p, tfp)
            bnd_constr_hat = _bnd_constr_hat

        path_constr_hat: Optional[Callable[..., Any]] = None
        if self.path_constr is not None:
            def _path_constr_hat(xh, uh, ph, t_hat_or_t_phys):
                x, u, p, t = phys_from_scaled(xh, uh, ph, t_hat_or_t_phys)
                assert self.path_constr is not None
                return self.path_constr(x, u, p, t)
            path_constr_hat = _path_constr_hat

        state_constr_hat: Optional[Callable[..., Any]] = None
        if self.state_constr is not None:
            def _state_constr_hat(xh, ph, t_hat_or_t_phys):
                # no u needed
                x, _, p, t = phys_from_scaled(xh, ca.MX.zeros(self.n_u), ph, t_hat_or_t_phys)
                assert self.state_constr is not None
                return self.state_constr(x, p, t)
            state_constr_hat = _state_constr_hat

        def _scale_bounds(bounds, ref):
            if bounds is None:
                return None
            lb, ub = bounds
            return (np.asarray(lb, float).reshape(-1) / ref, np.asarray(ub, float).reshape(-1) / ref)

        # scale tf if time-normalized
        if t_ref is None:
            tf_hat: TfSpec = self.tf
        else:
            if isinstance(self.tf, tuple):
                lb, ub = self.tf
                tf_hat = (float(lb) / float(t_ref), float(ub) / float(t_ref))
            else:
                tf_hat = float(self.tf) / float(t_ref)

        return OCP(
            n_x=self.n_x,
            n_u=self.n_u,
            n_p=self.n_p,
            t0=float(t0_hat),
            tf=tf_hat,
            f_dyn=f_dyn_hat,
            l_run=l_run_hat,
            l_end=l_end_hat,
            bnd_constr=bnd_constr_hat,
            path_constr=path_constr_hat,
            state_constr=state_constr_hat,
            x_bounds=_scale_bounds(self.x_bounds, x_ref),
            u_bounds=_scale_bounds(self.u_bounds, u_ref),
            p_bounds=None if self.n_p == 0 else _scale_bounds(self.p_bounds, p_ref),
            x0_fixed=None if self.x0_fixed is None else (self.x0_fixed / x_ref),
            p0_guess=None if self.p0_guess is None else (self.p0_guess / p_ref),
            scaling=scaling_hat,
        )

__all__ = ["OCP", "TfSpec"]

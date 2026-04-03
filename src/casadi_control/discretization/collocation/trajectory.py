"""Trajectory evaluators for direct-collocation solutions.

The evaluators expose interpolation-based access to primal and dual quantities
at arbitrary times.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..base import Array, TimeLike, Trajectory, DualTrajectory
from .common import validate_s_mesh
from .decode import CollocationPrimalGrid, CollocationAdjointGrid


# =============================================================================
# Barycentric interpolation utilities
# =============================================================================

def _barycentric_weights(nodes: Array) -> Array:
    """Compute first-form barycentric interpolation weights."""
    x = np.asarray(nodes, float).reshape(-1)
    m = x.size
    w = np.ones(m, dtype=float)
    for j in range(m):
        for k in range(m):
            if k != j:
                denom = x[j] - x[k]
                if abs(float(denom)) <= 1e-16:
                    raise ValueError("Interpolation nodes must be distinct")
                w[j] /= denom
    return w


def _barycentric_eval(nodes: Array, w: Array, vals: Array, xq: Array) -> Array:
    """Evaluate barycentric interpolant at query points."""
    nodes = np.asarray(nodes, float).reshape(-1)
    w = np.asarray(w, float).reshape(-1)

    vals = np.asarray(vals, float)
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    xq = np.asarray(xq, float).reshape(-1)

    m = nodes.size
    if w.size != m:
        raise ValueError("weights and nodes length mismatch")
    if vals.shape[0] != m:
        raise ValueError("vals must have the same number of rows as nodes")

    q = xq.size
    d = vals.shape[1]
    out = np.empty((q, d), dtype=float)

    for i in range(q):
        xi = xq[i]
        diff = xi - nodes

        hit = np.where(np.abs(diff) <= 1e-14)[0]
        if hit.size > 0:
            out[i, :] = vals[int(hit[0]), :]
            continue

        tmp = w / diff
        denom = float(np.sum(tmp))
        numer = tmp @ vals
        out[i, :] = numer / denom

    return out


def _uniform_s_mesh(N: int) -> np.ndarray:
    """Return a uniform normalized mesh of length ``N + 1``."""
    return np.linspace(0.0, 1.0, N + 1, dtype=float)


def _infer_s_mesh_from_t_mesh(t_mesh: np.ndarray, *, t0: float, tf: float) -> Optional[np.ndarray]:
    """
    If t_mesh looks like t0 + tf * s_mesh, infer s_mesh = (t_mesh - t0)/tf.
    Returns None if tf <= 0.
    """
    tf = float(tf)
    if tf <= 0.0:
        return None
    t_mesh = np.asarray(t_mesh, float).reshape(-1)
    s = (t_mesh - float(t0)) / tf
    return s


def _interval_index_and_local_s(
    *,
    t: np.ndarray,
    t0: float,
    tf: float,
    s_mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map physical (layout-coordinate) time t to interval index i and local coordinate s_hat in [0,1].

    Uses s_global = (t - t0)/tf in [0,1], then finds i such that
        s_mesh[i] <= s_global <= s_mesh[i+1]
    and sets
        s_hat = (s_global - s_mesh[i]) / (s_mesh[i+1] - s_mesh[i]).
    """
    t = np.asarray(t, float).reshape(-1)
    tf = float(tf)
    if tf <= 0.0:
        raise ValueError("tf must be positive to evaluate trajectories")

    # Clamp to [t0, t0+tf]
    tmin = float(t0)
    tmax = float(t0) + tf
    tt = np.clip(t, tmin, tmax)

    # Global normalized coordinate in [0,1]
    s_global = (tt - float(t0)) / tf

    # Interval index by searching in s_mesh
    # searchsorted gives insertion point; subtract 1 to get left interval.
    i = np.searchsorted(s_mesh, s_global, side="right") - 1
    N = s_mesh.size - 1
    i = np.clip(i, 0, N - 1)

    # Local coordinate on [0,1]
    s0 = s_mesh[i]
    s1 = s_mesh[i + 1]
    denom = (s1 - s0)
    # denom should be > 0 by validation
    s_hat = (s_global - s0) / denom

    # Exact endpoint: force to last interval, s_hat=1
    at_tf = np.isclose(tt, tmax)
    s_hat = np.where(at_tf, 1.0, s_hat)
    i = np.where(at_tf, N - 1, i)

    # Numerical guard
    s_hat = np.clip(s_hat, 0.0, 1.0)

    return i.astype(int), s_hat.astype(float)


# =============================================================================
# Primal trajectory evaluator
# =============================================================================

class CollocationPrimalTrajectory(Trajectory):
    """
    Evaluator for a direct-collocation primal solution.
    """

    def __init__(
        self,
        primal: CollocationPrimalGrid,
        tau: Array,
        tf: float,
        *,
        t0: Optional[float] = None,
        s_mesh: Optional[Array] = None,
    ) -> None:
        self.primal = primal
        self.tau = np.asarray(tau, float).reshape(-1)
        self.tf = float(tf)

        self.nx = int(primal.x_mesh.shape[1])
        self.nu = int(primal.u_colloc.shape[2])

        if self.tau.size != int(primal.x_colloc.shape[1]) + 1:
            raise ValueError("tau must have size K+1, where K = x_colloc.shape[1]")

        if t0 is None:
            self.t0 = float(np.asarray(primal.t_mesh, float).reshape(-1)[0])
        else:
            self.t0 = float(t0)

        N = int(primal.t_mesh.size - 1)

        if s_mesh is None:
            s_inf = _infer_s_mesh_from_t_mesh(primal.t_mesh, t0=self.t0, tf=self.tf)
            if s_inf is not None and s_inf.size == N + 1 and np.all(np.diff(s_inf) > 0.0):
                s_mesh_arr = s_inf.copy()
                s_mesh_arr[0] = 0.0
                s_mesh_arr[-1] = 1.0
            else:
                s_mesh_arr = _uniform_s_mesh(N)
        else:
            s_mesh_arr = np.asarray(s_mesh, float).reshape(-1)

        self.s_mesh = validate_s_mesh(s_mesh_arr, expected_size=N + 1)

        self._tau_x = self.tau
        self._w_x = _barycentric_weights(self._tau_x)

        self._tau_u = self.tau[1:]
        self._w_u = _barycentric_weights(self._tau_u)

    def _interval_index_and_local_tau(self, t: Array) -> tuple[Array, Array]:
        i, s_hat = _interval_index_and_local_s(
            t=np.asarray(t, float).reshape(-1),
            t0=self.t0,
            tf=self.tf,
            s_mesh=self.s_mesh,
        )
        return i, s_hat

    def x(self, t: TimeLike) -> Array:
        t = np.asarray(t, float).reshape(-1)
        i, s = self._interval_index_and_local_tau(t)

        _, K, nx = self.primal.x_colloc.shape

        out = np.zeros((t.size, nx), dtype=float)
        for ii in np.unique(i):
            mask = (i == ii)
            vals = np.vstack(
                [
                    self.primal.x_mesh[ii, :].reshape(1, nx),
                    self.primal.x_colloc[ii, :, :].reshape(K, nx),
                ]
            )
            out[mask, :] = _barycentric_eval(self._tau_x, self._w_x, vals, s[mask])
        return out

    def u(self, t: TimeLike) -> Array:
        t = np.asarray(t, float).reshape(-1)
        i, s = self._interval_index_and_local_tau(t)

        _, K, nu = self.primal.u_colloc.shape

        out = np.zeros((t.size, nu), dtype=float)
        for ii in np.unique(i):
            mask = (i == ii)
            vals = self.primal.u_colloc[ii, :, :].reshape(K, nu)
            out[mask, :] = _barycentric_eval(self._tau_u, self._w_u, vals, s[mask])
        return out


# =============================================================================
# Dual trajectory evaluator
# =============================================================================

class CollocationDualTrajectory(DualTrajectory):
    """
    Evaluator for costate and multiplier trajectories.
    """

    def __init__(
        self,
        dual: CollocationAdjointGrid,
        tau: Array,
        tf: float,
        *,
        t0: Optional[float] = None,
        s_mesh: Optional[Array] = None,
    ) -> None:
        self.dual = dual
        self.tau = np.asarray(tau, float).reshape(-1)
        self.tf = float(tf)

        self.nx = int(dual.costate_mesh.shape[-1])
        self.nc = int(dual.path_multiplier_colloc.shape[-1]) if dual.path_multiplier_colloc is not None else 0
        self.ns = int(dual.state_multiplier_colloc.shape[-1]) if dual.state_multiplier_colloc is not None else 0

        if self.tau.size != int(dual.t_colloc.shape[1]) + 1:
            raise ValueError("tau must have size K+1, where K = t_colloc.shape[1]")

        if t0 is None:
            self.t0 = float(np.asarray(dual.t_mesh, float).reshape(-1)[0])
        else:
            self.t0 = float(t0)

        N = int(dual.t_mesh.size - 1)

        if s_mesh is None:
            s_inf = _infer_s_mesh_from_t_mesh(dual.t_mesh, t0=self.t0, tf=self.tf)
            if s_inf is not None and s_inf.size == N + 1 and np.all(np.diff(s_inf) > 0.0):
                s_mesh_arr = s_inf.copy()
                s_mesh_arr[0] = 0.0
                s_mesh_arr[-1] = 1.0
            else:
                s_mesh_arr = _uniform_s_mesh(N)
        else:
            s_mesh_arr = np.asarray(s_mesh, float).reshape(-1)

        self.s_mesh = validate_s_mesh(s_mesh_arr, expected_size=N + 1)

        self._tau_costate = self.tau
        self._w_costate = _barycentric_weights(self._tau_costate)

        self._tau_multiplier = self.tau[1:]
        self._w_multiplier = _barycentric_weights(self._tau_multiplier)

    def _interval_index_and_local_tau(self, t: Array) -> tuple[Array, Array]:
        i, s_hat = _interval_index_and_local_s(
            t=np.asarray(t, float).reshape(-1),
            t0=self.t0,
            tf=self.tf,
            s_mesh=self.s_mesh,
        )
        return i, s_hat

    def costate(self, t: TimeLike) -> Array:
        t = np.asarray(t, float).reshape(-1)
        i, s = self._interval_index_and_local_tau(t)

        _, K, nx = self.dual.costate_colloc.shape

        out = np.zeros((t.size, nx), dtype=float)
        for ii in np.unique(i):
            mask = (i == ii)
            vals = np.vstack(
                [
                    self.dual.costate_mesh[ii, :].reshape(1, nx),
                    self.dual.costate_colloc[ii, :, :].reshape(K, nx),
                ]
            )
            out[mask, :] = _barycentric_eval(self._tau_costate, self._w_costate, vals, s[mask])
        return out

    def path_multiplier(self, t: TimeLike) -> Array:
        if self.dual.path_multiplier_colloc is None:
            return np.zeros((np.asarray(t).reshape(-1).size, 0), dtype=float)

        t = np.asarray(t, float).reshape(-1)
        i, s = self._interval_index_and_local_tau(t)

        _, K, nc = self.dual.path_multiplier_colloc.shape

        out = np.zeros((t.size, nc), dtype=float)
        for ii in np.unique(i):
            mask = (i == ii)
            vals = self.dual.path_multiplier_colloc[ii, :, :].reshape(K, nc)
            out[mask, :] = _barycentric_eval(self._tau_multiplier, self._w_multiplier, vals, s[mask])
        return out

    def state_multiplier(self, t: TimeLike) -> Array:
        if self.dual.state_multiplier_colloc is None:
            return np.zeros((np.asarray(t).reshape(-1).size, 0), dtype=float)

        t = np.asarray(t, float).reshape(-1)
        i, s = self._interval_index_and_local_tau(t)

        _, K, ns = self.dual.state_multiplier_colloc.shape

        out = np.zeros((t.size, ns), dtype=float)
        for ii in np.unique(i):
            mask = (i == ii)
            vals = self.dual.state_multiplier_colloc[ii, :, :].reshape(K, ns)
            out[mask, :] = _barycentric_eval(self._tau_multiplier, self._w_multiplier, vals, s[mask])
        return out

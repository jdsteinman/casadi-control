"""Initial-guess construction for direct collocation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..._array_utils import as_sized_1d_float_vector
from ..base import Guess, NLPLike, Trajectory
from .common import is_scaled_meta
from .decode import CollocationMeta, CollocationVarIndex


def _set_block(w0: np.ndarray, idx_pair, value: np.ndarray) -> None:
    """Write one contiguous decision-vector block in-place."""
    a, b = idx_pair
    v = np.asarray(value, float).reshape(-1)
    if (b - a) != v.size:
        raise ValueError(f"Block size mismatch: idx {a}:{b} expects {b-a}, got {v.size}")
    w0[a:b] = v


def _require_var_index(meta: CollocationMeta) -> CollocationVarIndex:
    """Return validated variable-index map from NLP metadata."""
    vix = meta.var_index()
    return vix


def _pack_blocks(nlp: NLPLike, w0: np.ndarray, blocks: Dict[str, Any]) -> np.ndarray:
    """Pack structured state/control/time/parameter arrays into flat ``w0``."""
    meta = CollocationMeta(nlp.meta or {})
    vix = _require_var_index(meta)

    w0 = np.asarray(w0, float).reshape(-1).copy()

    if "X_mesh" in blocks:
        X_mesh = np.asarray(blocks["X_mesh"], float)
        for i, (a, b) in enumerate(vix.X_mesh):
            _set_block(w0, (a, b), X_mesh[i, :])

    if "X_colloc" in blocks:
        Xc = np.asarray(blocks["X_colloc"], float)
        Xc_idx = vix.X_colloc
        N = len(Xc_idx)
        K = len(Xc_idx[0]) if N > 0 else 0
        for i in range(N):
            for j in range(K):
                _set_block(w0, Xc_idx[i][j], Xc[i, j, :])

    if "U_colloc" in blocks:
        Uc = np.asarray(blocks["U_colloc"], float)
        Uc_idx = vix.U_colloc
        N = len(Uc_idx)
        K = len(Uc_idx[0]) if N > 0 else 0
        for i in range(N):
            for j in range(K):
                _set_block(w0, Uc_idx[i][j], Uc[i, j, :])

    if "tf" in blocks and vix.tf is not None:
        _set_block(w0, vix.tf, np.array([float(blocks["tf"])], dtype=float))

    if "p" in blocks and vix.p is not None:
        _set_block(w0, vix.p, np.asarray(blocks["p"], float).reshape(-1))

    return w0


# =============================================================================
# Scaling helpers (physical <-> solver)
# =============================================================================

def _tf_phys_to_solver(meta: CollocationMeta, tf_phys: float) -> float:
    """Map physical final-time value to solver-space coordinate."""
    if not is_scaled_meta(meta):
        return float(tf_phys)
    t_ref = meta.t_ref
    if t_ref is None:
        raise RuntimeError("meta['t_ref'] must be set when meta['space']=='scaled'")
    return float(tf_phys) / float(t_ref)


def _blocks_phys_to_solver(meta: CollocationMeta, blocks_phys: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PHYSICAL blocks to SOLVER blocks using meta['scaling'] and meta['space'].

    If meta['space'] == 'physical', this is identity.
    If meta['space'] == 'scaled', divides x/u/p by refs, and divides tf by t_ref.
    """
    out: Dict[str, Any] = dict(blocks_phys)

    if not is_scaled_meta(meta):
        out["tf"] = float(out["tf"])
        return out

    s = meta.scaling
    x_ref = np.asarray(s.x_ref, float).reshape(1, -1) if getattr(s, "x_ref", None) is not None else None
    u_ref = np.asarray(s.u_ref, float).reshape(1, 1, -1) if getattr(s, "u_ref", None) is not None else None
    p_ref = np.asarray(s.p_ref, float).reshape(-1) if getattr(s, "p_ref", None) is not None else None

    if x_ref is not None:
        if "X_mesh" in out:
            out["X_mesh"] = np.asarray(out["X_mesh"], float) / x_ref
        if "X_colloc" in out:
            out["X_colloc"] = np.asarray(out["X_colloc"], float) / x_ref.reshape(1, 1, -1)

    if u_ref is not None and "U_colloc" in out:
        out["U_colloc"] = np.asarray(out["U_colloc"], float) / u_ref

    if p_ref is not None and "p" in out and out["p"] is not None:
        out["p"] = np.asarray(out["p"], float).reshape(-1) / p_ref

    out["tf"] = _tf_phys_to_solver(meta, float(out["tf"]))
    return out


# =============================================================================
# Decision-time sampling (PHYSICAL time for public API)
# =============================================================================

def _decision_times_solver_from_meta(meta: CollocationMeta, *, tf_solver: float) -> Dict[str, np.ndarray]:
    """
    Returns SOLVER/OCP decision times (t_mesh, t_colloc, t_col_flat).

    Uses meta['s_mesh'] (preferred) to support nonuniform meshes.
    Falls back to the old uniform mesh if s_mesh is not present.
    """
    layout = meta.layout
    times = layout.times_numpy(tf=float(tf_solver))
    return {
        "t_mesh": times["t_mesh"],
        "t_colloc": times["t_colloc"],
        "t_col_flat": times["t_colloc_flat"],
    }


def _decision_times_physical(meta: CollocationMeta, *, tf_phys: float) -> Dict[str, np.ndarray]:
    """
    Returns PHYSICAL decision times (t_mesh, t_colloc, t_col_flat) used for sampling.
    """
    if not is_scaled_meta(meta):
        # layout operates in physical time directly (solver time == physical time)
        return _decision_times_solver_from_meta(meta, tf_solver=float(tf_phys))

    # scaled space: decision times are solver-time, then map to physical
    tf_solver = _tf_phys_to_solver(meta, float(tf_phys))
    times_solver = _decision_times_solver_from_meta(meta, tf_solver=float(tf_solver))

    t_ref = float(meta.t_ref)
    t0_phys = meta.t0_phys

    out = dict(times_solver)
    out["t_mesh"] = t0_phys + t_ref * out["t_mesh"]
    out["t_colloc"] = t0_phys + t_ref * out["t_colloc"]
    out["t_col_flat"] = out["t_colloc"].reshape(-1)
    return out


# =============================================================================
# Structured block builders (PUBLIC inputs are PHYSICAL)
# =============================================================================

def _blocks_from_constant(
    nlp: NLPLike,
    *,
    tf_phys: float,
    x_const: Any,
    u_const: Any,
) -> Dict[str, Any]:
    """Build physical-space guess blocks from constant state/control values."""
    meta = CollocationMeta(nlp.meta or {})
    nx = meta.n_x
    nu = meta.n_u

    x_vec = as_sized_1d_float_vector(x_const, nx, name="x", broadcast_scalar=True)
    u_vec = as_sized_1d_float_vector(u_const, nu, name="u", broadcast_scalar=True)

    return _blocks_from_functions_physical(
        nlp,
        tf_phys=float(tf_phys),
        x=lambda _t, xv=x_vec: xv,
        u=lambda _t, uv=u_vec: uv,
    )


def _blocks_from_functions_physical(
    nlp: NLPLike,
    *,
    tf_phys: float,
    x: Optional[Callable[[float], Any]] = None,
    u: Optional[Callable[[float], Any]] = None,
) -> Dict[str, Any]:
    """Sample user-supplied ``x(t)``, ``u(t)`` functions on decision times."""
    meta = CollocationMeta(nlp.meta or {})
    N = meta.N
    K = meta.K
    nx = meta.n_x
    nu = meta.n_u

    times = _decision_times_physical(meta, tf_phys=tf_phys)
    t_mesh = times["t_mesh"]
    t_colloc = times["t_colloc"]

    blocks: Dict[str, Any] = {"tf": float(tf_phys)}

    if x is not None:
        X_mesh = np.zeros((N + 1, nx), dtype=float)
        for i in range(N + 1):
            X_mesh[i, :] = as_sized_1d_float_vector(
                x(float(t_mesh[i])),
                nx,
                name="x",
                broadcast_scalar=True,
            )

        X_colloc = np.zeros((N, K, nx), dtype=float)
        for i in range(N):
            for j in range(K):
                X_colloc[i, j, :] = as_sized_1d_float_vector(
                    x(float(t_colloc[i, j])),
                    nx,
                    name="x",
                    broadcast_scalar=True,
                )

        blocks["X_mesh"] = X_mesh
        blocks["X_colloc"] = X_colloc

    if u is not None:
        U_colloc = np.zeros((N, K, nu), dtype=float)
        for i in range(N):
            for j in range(K):
                U_colloc[i, j, :] = as_sized_1d_float_vector(
                    u(float(t_colloc[i, j])),
                    nu,
                    name="u",
                    broadcast_scalar=True,
                )
        blocks["U_colloc"] = U_colloc

    return blocks


def _blocks_from_prev_trajectory_physical(
    nlp: NLPLike,
    *,
    tf_phys: float,
    prev: Trajectory,
) -> Dict[str, Any]:
    """Sample a previous trajectory object on current decision-time grid."""
    meta = CollocationMeta(nlp.meta or {})
    N = meta.N
    K = meta.K
    nx = meta.n_x
    nu = meta.n_u

    times = _decision_times_physical(meta, tf_phys=tf_phys)
    t_mesh = times["t_mesh"]
    t_colloc = times["t_colloc"]

    blocks: Dict[str, Any] = {"tf": float(tf_phys)}

    X_mesh = prev.x(t_mesh)
    if X_mesh.shape != (N + 1, nx):
        raise ValueError(f"prev.x(t_mesh) returned {X_mesh.shape}, expected {(N+1, nx)}")

    X_colloc = prev.x(t_colloc.reshape(-1)).reshape(N, K, nx)
    U_colloc = prev.u(t_colloc.reshape(-1)).reshape(N, K, nu)

    blocks["X_mesh"] = np.asarray(X_mesh, float)
    blocks["X_colloc"] = np.asarray(X_colloc, float)
    blocks["U_colloc"] = np.asarray(U_colloc, float)

    return blocks


# =============================================================================
# Public API
# =============================================================================

def guess_collocation(
    nlp: NLPLike,
    *,
    strategy: str = "default",
    prev: Optional[Trajectory] = None,
    **kwargs: Any,
) -> Guess:
    """
    Construct an initial guess for a direct collocation NLP.

    Conventions
    -----------
    - For strategies: const, functions, prev
      inputs are PHYSICAL and are converted to SOLVER if meta['space']=='scaled'.

    - For strategy: blocks
      blocks are SOLVER-space (no backwards-compat toggle).

    Returns
    -------
    Guess
        Flat decision-vector guess with strategy diagnostics in ``info``.
    """
    meta = CollocationMeta(nlp.meta or {})
    if (meta.meta.get("discretization") or "").lower() != "direct_collocation":
        raise RuntimeError("guess_collocation supports direct_collocation only")

    w0 = np.asarray(kwargs.get("w0", nlp.w0), float).reshape(-1).copy()
    info: Dict[str, Any] = {"strategy": strategy}

    if strategy in ("default", "nlp"):
        return Guess(w0=w0, info=info)

    if strategy == "blocks":
        blocks = kwargs["blocks"]
        if not isinstance(blocks, dict):
            raise ValueError("strategy='blocks' requires blocks={<key>: array, ...} (SOLVER space)")
        w0 = _pack_blocks(nlp, w0, dict(blocks))
        info["used"] = sorted(list(blocks.keys()))
        info["space_in"] = "solver"
        info["space_packed"] = "solver"
        return Guess(w0=w0, info=info)

    if strategy == "const":
        tf_phys = float(kwargs["tf"])
        x_const = kwargs["x"]
        u_const = kwargs["u"]

        blocks_phys = _blocks_from_constant(nlp, tf_phys=tf_phys, x_const=x_const, u_const=u_const)
        if "p" in kwargs and kwargs["p"] is not None:
            blocks_phys["p"] = kwargs["p"]

        blocks_solver = _blocks_phys_to_solver(meta, blocks_phys)
        w0 = _pack_blocks(nlp, w0, blocks_solver)

        info["used"] = sorted(blocks_phys.keys())
        info["space_in"] = "physical"
        info["space_packed"] = "solver"
        return Guess(w0=w0, info=info)

    if strategy == "functions":
        tf_phys = float(kwargs["tf"])
        blocks_phys = _blocks_from_functions_physical(
            nlp,
            tf_phys=tf_phys,
            x=kwargs.get("x", None),
            u=kwargs.get("u", None),
        )
        if "p" in kwargs and kwargs["p"] is not None:
            blocks_phys["p"] = kwargs["p"]

        blocks_solver = _blocks_phys_to_solver(meta, blocks_phys)
        w0 = _pack_blocks(nlp, w0, blocks_solver)

        info["used"] = sorted(blocks_phys.keys())
        info["space_in"] = "physical"
        info["space_packed"] = "solver"
        return Guess(w0=w0, info=info)

    if strategy == "prev":
        if prev is None:
            raise ValueError("strategy='prev' requires prev=<Trajectory>")
        tf_phys = float(kwargs.get("tf", prev.tf))

        blocks_phys = _blocks_from_prev_trajectory_physical(nlp, tf_phys=tf_phys, prev=prev)
        if "p" in kwargs and kwargs["p"] is not None:
            blocks_phys["p"] = kwargs["p"]

        blocks_solver = _blocks_phys_to_solver(meta, blocks_phys)
        w0 = _pack_blocks(nlp, w0, blocks_solver)

        info["used"] = sorted(blocks_phys.keys())
        info["space_in"] = "physical"
        info["space_packed"] = "solver"

        return Guess(
            w0=w0,
            mult_x0=kwargs.get("mult_x0", None),
            mult_g0=kwargs.get("mult_g0", None),
            info=info,
        )

    raise ValueError(f"Unknown guess strategy: {strategy!r}")

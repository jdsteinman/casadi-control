"""Initial-guess construction for direct collocation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..base import Guess, NLPLike, Trajectory


# =============================================================================
# Helpers: vectorization and packing
# =============================================================================

def _as_vec(val: Any, n: int) -> np.ndarray:
    """Convert scalar/vector input to a dense vector of length ``n``."""
    arr = np.asarray(val, float).reshape(-1)
    if arr.size == 1 and n != 1:
        return np.full((n,), float(arr[0]))
    if arr.size != n:
        raise ValueError(f"Expected size {n}, got {arr.size}")
    return arr


def _set_block(w0: np.ndarray, idx_pair, value: np.ndarray) -> None:
    """Write one contiguous decision-vector block in-place."""
    a, b = idx_pair
    v = np.asarray(value, float).reshape(-1)
    if (b - a) != v.size:
        raise ValueError(f"Block size mismatch: idx {a}:{b} expects {b-a}, got {v.size}")
    w0[a:b] = v


def _require_var_index(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Return validated variable-index map from NLP metadata."""
    vix = meta["var_index"]
    if not vix:
        raise RuntimeError("nlp.meta['var_index'] missing; cannot pack guess blocks")
    if not isinstance(vix, dict):
        raise TypeError("nlp.meta['var_index'] must be a dict")
    return vix


def _pack_blocks(nlp: NLPLike, w0: np.ndarray, blocks: Dict[str, Any]) -> np.ndarray:
    """Pack structured state/control/time/parameter arrays into flat ``w0``."""
    meta = nlp.meta or {}
    vix = _require_var_index(meta)

    w0 = np.asarray(w0, float).reshape(-1).copy()

    if "X_mesh" in blocks:
        X_mesh = np.asarray(blocks["X_mesh"], float)
        for i, (a, b) in enumerate(vix["X_mesh"]):
            _set_block(w0, (a, b), X_mesh[i, :])

    if "X_colloc" in blocks:
        Xc = np.asarray(blocks["X_colloc"], float)
        Xc_idx = vix["X_colloc"]
        N = len(Xc_idx)
        K = len(Xc_idx[0]) if N > 0 else 0
        for i in range(N):
            for j in range(K):
                _set_block(w0, Xc_idx[i][j], Xc[i, j, :])

    if "U_colloc" in blocks:
        Uc = np.asarray(blocks["U_colloc"], float)
        Uc_idx = vix["U_colloc"]
        N = len(Uc_idx)
        K = len(Uc_idx[0]) if N > 0 else 0
        for i in range(N):
            for j in range(K):
                _set_block(w0, Uc_idx[i][j], Uc[i, j, :])

    if "tf" in blocks and vix.get("tf", None) is not None:
        _set_block(w0, vix["tf"], np.array([float(blocks["tf"])], dtype=float))

    if "p" in blocks and vix.get("p", None) is not None:
        _set_block(w0, vix["p"], np.asarray(blocks["p"], float).reshape(-1))

    return w0


# =============================================================================
# Scaling helpers (physical <-> solver)
# =============================================================================

def _is_scaled(meta: Dict[str, Any]) -> bool:
    """Return whether NLP metadata indicates scaled solver coordinates."""
    return str(meta["space"]).lower() == "scaled"


def _tf_phys_to_solver(meta: Dict[str, Any], tf_phys: float) -> float:
    """Map physical final-time value to solver-space coordinate."""
    if not _is_scaled(meta):
        return float(tf_phys)
    t_ref = meta["t_ref"]
    if t_ref is None:
        raise RuntimeError("meta['t_ref'] must be set when meta['space']=='scaled'")
    return float(tf_phys) / float(t_ref)


def _blocks_phys_to_solver(meta: Dict[str, Any], blocks_phys: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert PHYSICAL blocks to SOLVER blocks using meta['scaling'] and meta['space'].

    If meta['space'] == 'physical', this is identity.
    If meta['space'] == 'scaled', divides x/u/p by refs, and divides tf by t_ref.
    """
    out: Dict[str, Any] = dict(blocks_phys)

    if not _is_scaled(meta):
        out["tf"] = float(out["tf"])
        return out

    s = meta["scaling"]
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

def _decision_times_solver_from_meta(meta: Dict[str, Any], *, tf_solver: float) -> Dict[str, np.ndarray]:
    """
    Returns SOLVER/OCP decision times (t_mesh, t_colloc, t_col_flat).

    Uses meta['s_mesh'] (preferred) to support nonuniform meshes.
    Falls back to the old uniform mesh if s_mesh is not present.
    """
    t0 = float(meta.get("t0_ocp", 0.0))
    N = int(meta["N"])
    K = int(meta["degree"])
    tau = np.asarray(meta["tau"], float).reshape(-1)  # length K+1 typically

    if "s_mesh" in meta and meta["s_mesh"] is not None:
        s_mesh = np.asarray(meta["s_mesh"], float).reshape(-1)
        if s_mesh.size != N + 1:
            raise ValueError(f"meta['s_mesh'] has length {s_mesh.size}, expected {N+1}")
        if not np.all(np.diff(s_mesh) > 0.0):
            raise ValueError("meta['s_mesh'] must be strictly increasing")
        if abs(float(s_mesh[0]) - 0.0) > 1e-10 or abs(float(s_mesh[-1]) - 1.0) > 1e-10:
            raise ValueError("meta['s_mesh'] must start at 0 and end at 1")

        ds = np.diff(s_mesh)  # length N

        t_mesh = t0 + float(tf_solver) * s_mesh

        t_colloc = np.zeros((N, K), dtype=float)
        for i in range(N):
            s_i = float(s_mesh[i])
            ds_i = float(ds[i])
            for j in range(K):
                s_ij = s_i + ds_i * float(tau[j + 1])
                t_colloc[i, j] = t0 + float(tf_solver) * s_ij

        return {
            "t_mesh": t_mesh,
            "t_colloc": t_colloc,
            "t_col_flat": t_colloc.reshape(-1),
        }

    # -------------------------------------------------------------------------
    # Backwards-compatible fallback: uniform mesh
    # -------------------------------------------------------------------------
    # Old convention:
    #   t = t0 + tf * ((i + tau)/N)
    t_mesh = t0 + float(tf_solver) * (np.arange(N + 1, dtype=float) / float(N))

    t_colloc = np.zeros((N, K), dtype=float)
    for i in range(N):
        for j in range(K):
            t_colloc[i, j] = t0 + float(tf_solver) * ((float(i) + float(tau[j + 1])) / float(N))

    return {
        "t_mesh": t_mesh,
        "t_colloc": t_colloc,
        "t_col_flat": t_colloc.reshape(-1),
    }


def _decision_times_physical(meta: Dict[str, Any], *, tf_phys: float) -> Dict[str, np.ndarray]:
    """
    Returns PHYSICAL decision times (t_mesh, t_colloc, t_col_flat) used for sampling.
    """
    if not _is_scaled(meta):
        # layout operates in physical time directly (solver time == physical time)
        return _decision_times_solver_from_meta(meta, tf_solver=float(tf_phys))

    # scaled space: decision times are solver-time, then map to physical
    tf_solver = _tf_phys_to_solver(meta, float(tf_phys))
    times_solver = _decision_times_solver_from_meta(meta, tf_solver=float(tf_solver))

    t_ref = float(meta["t_ref"])
    t0_phys = float(meta["t0_phys"])

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
    meta = nlp.meta or {}
    nx = int(meta["n_x"])
    nu = int(meta["n_u"])

    x_vec = _as_vec(x_const, nx)
    u_vec = _as_vec(u_const, nu)

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
    meta = nlp.meta or {}
    N = int(meta["N"])
    K = int(meta["degree"])
    nx = int(meta["n_x"])
    nu = int(meta["n_u"])

    times = _decision_times_physical(meta, tf_phys=tf_phys)
    t_mesh = times["t_mesh"]
    t_colloc = times["t_colloc"]

    blocks: Dict[str, Any] = {"tf": float(tf_phys)}

    if x is not None:
        X_mesh = np.zeros((N + 1, nx), dtype=float)
        for i in range(N + 1):
            X_mesh[i, :] = _as_vec(x(float(t_mesh[i])), nx)

        X_colloc = np.zeros((N, K, nx), dtype=float)
        for i in range(N):
            for j in range(K):
                X_colloc[i, j, :] = _as_vec(x(float(t_colloc[i, j])), nx)

        blocks["X_mesh"] = X_mesh
        blocks["X_colloc"] = X_colloc

    if u is not None:
        U_colloc = np.zeros((N, K, nu), dtype=float)
        for i in range(N):
            for j in range(K):
                U_colloc[i, j, :] = _as_vec(u(float(t_colloc[i, j])), nu)
        blocks["U_colloc"] = U_colloc

    return blocks


def _blocks_from_prev_trajectory_physical(
    nlp: NLPLike,
    *,
    tf_phys: float,
    prev: Trajectory,
) -> Dict[str, Any]:
    """Sample a previous trajectory object on current decision-time grid."""
    meta = nlp.meta or {}
    N = int(meta["N"])
    K = int(meta["degree"])
    nx = int(meta["n_x"])
    nu = int(meta["n_u"])

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
    meta = nlp.meta or {}
    if (meta.get("discretization") or "").lower() != "direct_collocation":
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

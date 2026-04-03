"""Collocation NLP transcription.

Implements conversion of an :class:`~casadi_control.problem.ocp.OCP` into a CasADi
NLP using direct collocation on a normalized mesh ``s in [0, 1]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np

from ..base import NLP
from .schemes import CollocationTable
from .decode import CollocationLayout


def _validate_s_mesh(s_mesh: np.ndarray) -> None:
    """Validate normalized mesh monotonicity and endpoint convention."""
    s_mesh = np.asarray(s_mesh, float).reshape(-1)
    if s_mesh.size < 2:
        raise ValueError("s_mesh must have length >= 2")
    if not np.all(np.diff(s_mesh) > 0.0):
        raise ValueError("s_mesh must be strictly increasing")
    if abs(float(s_mesh[0]) - 0.0) > 1e-12 or abs(float(s_mesh[-1]) - 1.0) > 1e-12:
        raise ValueError("s_mesh must start at 0 and end at 1 (within tolerance)")


def _midpoint_safe(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Midpoint of bounds, treating +/-inf as 0.0."""
    lb = np.asarray(lb, float).reshape(-1)
    ub = np.asarray(ub, float).reshape(-1)
    mid = 0.5 * (lb + ub)
    mid = np.where(np.isfinite(mid), mid, 0.0)
    return mid


def _safe_x_fill(ocp: Any, *, xlb: np.ndarray, xub: np.ndarray, n_x: int) -> np.ndarray:
    """Return a robust initial state guess from fixed value or midpoint bounds."""
    if getattr(ocp, "x0_fixed", None) is not None:
        x = np.asarray(ocp.x0_fixed, float).reshape(-1)
        if x.size != n_x:
            raise ValueError(f"ocp.x0_fixed has size {x.size}, expected {n_x}")
        return x
    x = _midpoint_safe(xlb, xub)
    if x.size != n_x:
        raise ValueError(f"x_fill has size {x.size}, expected {n_x}")
    return x


def _safe_u_fill(ocp: Any, *, n_u: int) -> np.ndarray:
    """Return a robust initial control guess consistent with control bounds."""
    u = np.zeros((n_u,), dtype=float)
    if getattr(ocp, "u_bounds", None) is not None and ocp.u_bounds is not None:
        ulb = np.asarray(ocp.u_bounds[0], float).reshape(-1)
        uub = np.asarray(ocp.u_bounds[1], float).reshape(-1)
        if ulb.size != n_u or uub.size != n_u:
            raise ValueError("u_bounds size mismatch")
        u = np.clip(u, ulb, uub)
    return u


def build_collocation_nlp(
    ocp: Any,
    *,
    s_mesh: np.ndarray,
    table: CollocationTable,
) -> NLP:
    """
    Build a direct collocation NLP with a (possibly nonuniform) normalized mesh.

    Mesh convention
    ---------------
    The user supplies `s_mesh` in [0,1], length N+1, strictly increasing.
    Physical/OCP time is then ``t(s) = t0 + tf * s``, where ``tf`` may be
    fixed or free (in the OCP's time coordinates).

    This keeps free-final-time compatible with arbitrary spacing patterns.

    Returns
    -------
    NLP
        Solver-ready nonlinear program with indexing metadata and trajectory
        decode function attached.
    """
    ocp.validate()

    s_mesh = np.asarray(s_mesh, float).reshape(-1)
    _validate_s_mesh(s_mesh)

    N = int(s_mesh.size - 1)
    ds = np.diff(s_mesh)  # length N, positive

    n_x, n_u, n_p = ocp.n_x, ocp.n_u, ocp.n_p

    tau = np.asarray(table.tau, float)
    D = table.D
    c = table.c
    wq = np.asarray(table.w, float)
    K = int(table.degree)

    layout = CollocationLayout(
        N=int(N),
        K=int(K),
        tau=tau,
        s_mesh=s_mesh,
        t0=float(getattr(ocp, "t0", 0.0)),
    )

    scaling = getattr(ocp, "scaling", None)

    # For trajectory reconstruction
    t_ref = None
    t0_phys = None
    if scaling is not None:
        if getattr(scaling, "t_ref", None) is not None:
            t_ref = float(scaling.t_ref)
            t0_phys = float(getattr(scaling, "t0_phys", 0.0))

    # Defect scaling vector in OCP coordinates
    if scaling is not None:
        defect_ref = np.asarray(scaling.defect_scale_vec(nx=n_x), float).reshape(-1)
    else:
        defect_ref = np.ones(n_x, dtype=float)

    # -------------------------------------------------------------------------
    # Time variables (in OCP's time coordinates)
    # -------------------------------------------------------------------------
    free_tf = isinstance(ocp.tf, tuple)
    if free_tf:
        tf_lb, tf_ub = ocp.tf
        tf = ca.MX.sym("tf")  # type: ignore[arg-type]
    else:
        tf = ca.MX(float(ocp.tf))

    # -------------------------------------------------------------------------
    # Parameters
    # -------------------------------------------------------------------------
    free_p = (n_p > 0) and (ocp.p_bounds is not None)
    if n_p > 0:
        if free_p:
            p = ca.MX.sym("p", n_p)  # type: ignore[arg-type]
        else:
            if ocp.p0_guess is None:
                raise ValueError("Fixed parameters require ocp.p0_guess")
            p = ca.DM(ocp.p0_guess)
    else:
        p = ca.DM([])

    # -------------------------------------------------------------------------
    # NLP containers
    # -------------------------------------------------------------------------
    w: List[ca.MX] = []
    w0: List[float] = []
    lbx: List[float] = []
    ubx: List[float] = []

    g: List[ca.MX] = []
    lbg: List[float] = []
    ubg: List[float] = []

    var_pos = 0
    con_pos = 0

    X_mesh_idx: List[Tuple[int, int]] = []
    X_col_idx: List[List[Tuple[int, int]]] = [[(0, 0) for _ in range(K)] for _ in range(N)]
    U_col_idx: List[List[Tuple[int, int]]] = [[(0, 0) for _ in range(K)] for _ in range(N)]
    tf_idx: Optional[Tuple[int, int]] = None
    p_idx: Optional[Tuple[int, int]] = None

    init_eq_idx: Optional[Tuple[int, int]] = None
    defect_idx: List[List[Tuple[int, int]]] = [[(0, 0) for _ in range(K)] for _ in range(N)]
    cont_idx: List[Tuple[int, int]] = []
    bnd_idx: Optional[Tuple[int, int]] = None
    path_idx: List[List[Optional[Tuple[int, int]]]] = [[None for _ in range(K)] for _ in range(N)]
    state_idx: List[List[Optional[Tuple[int, int]]]] = [[None for _ in range(K)] for _ in range(N)]

    # -------------------------------------------------------------------------
    # Initial state
    # -------------------------------------------------------------------------
    Xi = ca.MX.sym("X_0", n_x)  # type: ignore[arg-type]
    w += [Xi]
    X_mesh_idx.append((var_pos, var_pos + n_x))
    var_pos += n_x

    xlb = ocp.x_bounds[0] if ocp.x_bounds else -np.inf * np.ones(n_x)
    xub = ocp.x_bounds[1] if ocp.x_bounds else  np.inf * np.ones(n_x)
    lbx += list(np.asarray(xlb, float).reshape(-1))
    ubx += list(np.asarray(xub, float).reshape(-1))

    x_fill = _safe_x_fill(ocp, xlb=np.asarray(xlb, float), xub=np.asarray(xub, float), n_x=n_x)
    u_fill = _safe_u_fill(ocp, n_u=n_u)

    if ocp.x0_fixed is not None:
        w0 += list(np.asarray(ocp.x0_fixed, float).reshape(-1))
    else:
        w0 += list(x_fill)

    if ocp.x0_fixed is not None:
        r0 = (Xi - np.asarray(ocp.x0_fixed, float).reshape(-1)) / ca.DM(defect_ref)
        g += [r0]
        init_eq_idx = (con_pos, con_pos + n_x)
        con_pos += n_x
        lbg += [0.0] * n_x
        ubg += [0.0] * n_x

    X_mesh: List[ca.MX] = [Xi]
    X_colloc: List[List[ca.MX]] = []
    U_colloc: List[List[ca.MX]] = []

    # -------------------------------------------------------------------------
    # Objective accumulation
    # -------------------------------------------------------------------------
    J = 0

    f_dyn = ocp.f_dyn
    assert f_dyn is not None
    l_run = ocp.l_run
    l_end = ocp.l_end
    path_constr = ocp.path_constr
    state_constr = ocp.state_constr
    bnd_constr = ocp.bnd_constr

    # -------------------------------------------------------------------------
    # Per-interval variables and constraints
    # -------------------------------------------------------------------------
    for i in range(N):
        ds_i = float(ds[i])
        s_i = float(s_mesh[i])

        Xc: List[ca.MX] = []
        Uc: List[ca.MX] = []

        for j in range(K):
            Xij = ca.MX.sym(f"X_{i}_{j}", n_x)  # type: ignore[arg-type]
            Uij = ca.MX.sym(f"U_{i}_{j}", n_u)  # type: ignore[arg-type]
            Xc.append(Xij)
            Uc.append(Uij)

            w += [Xij]
            X_col_idx[i][j] = (var_pos, var_pos + n_x)
            var_pos += n_x

            w += [Uij]
            U_col_idx[i][j] = (var_pos, var_pos + n_u)
            var_pos += n_u

            xlb = ocp.x_bounds[0] if ocp.x_bounds else -np.inf * np.ones(n_x)
            xub = ocp.x_bounds[1] if ocp.x_bounds else  np.inf * np.ones(n_x)
            ulb = ocp.u_bounds[0] if ocp.u_bounds else -np.inf * np.ones(n_u)
            uub = ocp.u_bounds[1] if ocp.u_bounds else  np.inf * np.ones(n_u)

            lbx += list(np.asarray(xlb, float).reshape(-1))
            ubx += list(np.asarray(xub, float).reshape(-1))
            lbx += list(np.asarray(ulb, float).reshape(-1))
            ubx += list(np.asarray(uub, float).reshape(-1))

            w0 += list(x_fill) + list(u_fill)

        X_colloc.append(Xc)
        U_colloc.append(Uc)

        Xi_end = c[0] * Xi

        for j in range(K):
            xp = D[j, 0] * Xi
            for k in range(K):
                xp += D[j, k + 1] * Xc[k]

            # normalized node: s_ij = s_i + ds_i * tau[j+1]
            s_ij = s_i + ds_i * float(tau[j + 1])

            # OCP time coordinate: t_ij = t0 + tf * s_ij
            t_ij = float(ocp.t0) + tf * float(s_ij)

            fj = f_dyn(Xc[j], Uc[j], p, t_ij)

            # Defect: tf * ds_i * fj - xp = 0
            r_def = (tf * float(ds_i) * fj - xp) / ca.DM(defect_ref)
            g += [r_def]
            defect_idx[i][j] = (con_pos, con_pos + n_x)
            con_pos += n_x
            lbg += [0.0] * n_x
            ubg += [0.0] * n_x

            Xi_end += c[j + 1] * Xc[j]

            if l_run is not None:
                J += tf * float(ds_i) * float(wq[j]) * l_run(Xc[j], Uc[j], p, t_ij)

            if path_constr is not None:
                cj = path_constr(Xc[j], Uc[j], p, t_ij)
                m = int(cj.shape[0])
                if m > 0:
                    if scaling is not None:
                        pref = ca.DM(np.asarray(scaling.path_scale_vec(m=m), float))
                    else:
                        pref = ca.DM(np.ones(m))
                    g += [cj / pref]
                    path_idx[i][j] = (con_pos, con_pos + m)
                    con_pos += m
                    lbg += [-np.inf] * m
                    ubg += [0.0] * m

            if state_constr is not None:
                sj = state_constr(Xc[j], p, t_ij)
                m = int(sj.shape[0])
                if m > 0:
                    if scaling is not None:
                        sref = ca.DM(np.asarray(scaling.state_scale_vec(m=m), float))
                    else:
                        sref = ca.DM(np.ones(m))
                    g += [sj / sref]
                    state_idx[i][j] = (con_pos, con_pos + m)
                    con_pos += m
                    lbg += [-np.inf] * m
                    ubg += [0.0] * m

        Xi = ca.MX.sym(f"X_{i+1}", n_x)  # type: ignore[arg-type]
        w += [Xi]
        X_mesh_idx.append((var_pos, var_pos + n_x))
        var_pos += n_x

        xlb = ocp.x_bounds[0] if ocp.x_bounds else -np.inf * np.ones(n_x)
        xub = ocp.x_bounds[1] if ocp.x_bounds else  np.inf * np.ones(n_x)
        lbx += list(np.asarray(xlb, float).reshape(-1))
        ubx += list(np.asarray(xub, float).reshape(-1))
        w0 += [0.0] * n_x

        r_cont = (Xi_end - Xi) / ca.DM(defect_ref)
        g += [r_cont]
        cont_idx.append((con_pos, con_pos + n_x))
        con_pos += n_x
        lbg += [0.0] * n_x
        ubg += [0.0] * n_x

        X_mesh.append(Xi)

    # -------------------------------------------------------------------------
    # Terminal cost and boundary constraints (tf is in OCP coordinates)
    # -------------------------------------------------------------------------
    if l_end is not None:
        J += l_end(X_mesh[0], X_mesh[-1], p, ocp.t0, tf)

    if bnd_constr is not None:
        bval = bnd_constr(X_mesh[0], X_mesh[-1], p, ocp.t0, tf)
        nb = int(bval.shape[0])
        if nb > 0:
            if scaling is not None:
                bref = ca.DM(np.asarray(scaling.bnd_scale_vec(m=nb), float))
            else:
                bref = ca.DM(np.ones(nb))
            g += [bval / bref]
            bnd_idx = (con_pos, con_pos + nb)
            con_pos += nb
            lbg += [0.0] * nb
            ubg += [0.0] * nb

    # -------------------------------------------------------------------------
    # Free final time and free parameters
    # -------------------------------------------------------------------------
    if free_tf:
        w += [tf]
        tf_idx = (var_pos, var_pos + 1)
        var_pos += 1
        lbx += [float(tf_lb)]
        ubx += [float(tf_ub)]
        w0 += [0.5 * (float(tf_lb) + float(tf_ub))]

    if free_p:
        w += [p]
        p_idx = (var_pos, var_pos + n_p)
        var_pos += n_p
        lbx += list(np.asarray(ocp.p_bounds[0], float).reshape(-1))
        ubx += list(np.asarray(ocp.p_bounds[1], float).reshape(-1))
        w0 += list(np.asarray(ocp.p0_guess if ocp.p0_guess is not None else np.zeros(n_p), float).reshape(-1))

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------
    X_mesh_mat = ca.hcat(X_mesh).T  # (N+1, nx)

    X_col_list: List[ca.MX] = []
    U_col_list: List[ca.MX] = []
    for i in range(N):
        for j in range(K):
            X_col_list.append(X_colloc[i][j].T)
            U_col_list.append(U_colloc[i][j].T)

    X_col_mat = ca.vertcat(*X_col_list)  # (N*K, nx)
    U_col_mat = ca.vertcat(*U_col_list)  # (N*K, nu)

    # Times in OCP coordinates:
    #   t_mesh_hat = t0 + tf * s_mesh
    #   t_col_hat  = t0 + tf * (s_i + ds_i * tau[j+1])
    s_mesh_dm = ca.DM(list(s_mesh))
    t_mesh_hat = float(layout.t0) + tf * s_mesh_dm

    s_col = []
    for i in range(N):
        s_i = float(s_mesh[i])
        ds_i = float(ds[i])
        for j in range(K):
            s_col.append(s_i + ds_i * float(tau[j + 1]))
    t_col_hat = float(layout.t0) + tf * ca.DM(s_col)

    # Convert to physical time if scaling is active
    if scaling is not None and getattr(scaling, "space", "physical") == "scaled" and t_ref is not None:
        tf_phys = float(t_ref) * tf
        t_mesh = float(t0_phys) + float(t_ref) * t_mesh_hat
        t_col  = float(t0_phys) + float(t_ref) * t_col_hat
    else:
        tf_phys = tf
        t_mesh = t_mesh_hat
        t_col  = t_col_hat

    # -------------------------------------------------------------------------
    # NLP assembly
    # -------------------------------------------------------------------------
    wvec = ca.vertcat(*w)
    gvec = ca.vertcat(*g)
    prob = {"f": J, "x": wvec, "g": gvec}

    # Store bounds in BOTH spaces (OCP space and physical space) for plotting/archival
    x_bounds_ocp = ocp.x_bounds
    u_bounds_ocp = ocp.u_bounds
    p_bounds_ocp = ocp.p_bounds

    x_bounds_phys = x_bounds_ocp
    u_bounds_phys = u_bounds_ocp
    p_bounds_phys = p_bounds_ocp

    if scaling is not None and getattr(scaling, "x_ref", None) is not None:
        x_ref = np.asarray(scaling.x_ref, float)
        u_ref = np.asarray(scaling.u_ref, float) if getattr(scaling, "u_ref", None) is not None else None
        p_ref = np.asarray(scaling.p_ref, float) if getattr(scaling, "p_ref", None) is not None else None

        if x_bounds_ocp is not None:
            x_bounds_phys = scaling.unscale_bounds(x_bounds_ocp, ref=x_ref)
        if u_bounds_ocp is not None and u_ref is not None:
            u_bounds_phys = scaling.unscale_bounds(u_bounds_ocp, ref=u_ref)
        if p_bounds_ocp is not None and p_ref is not None:
            p_bounds_phys = scaling.unscale_bounds(p_bounds_ocp, ref=p_ref)

    meta: Dict[str, Any] = {
        "discretization": "direct_collocation",
        "N": int(N),
        "degree": int(K),
        "n_x": int(n_x),
        "n_u": int(n_u),
        "space": "scaled" if scaling is not None and getattr(scaling, "space", "physical") == "scaled" else "physical",
        "scaling": scaling,
        "t0_ocp": float(getattr(ocp, "t0", 0.0)),
        "t0_phys": float(t0_phys) if t0_phys is not None else float(getattr(ocp, "t0", 0.0)),
        "t_ref": float(t_ref) if t_ref is not None else None,
        "tau": tau,
        "s_mesh": np.asarray(s_mesh, float),
        "layout": layout,
        "quad_weights": wq,
        "bounds_ocp": {
            "x": x_bounds_ocp,
            "u": u_bounds_ocp,
            "p": p_bounds_ocp,
        },
        "bounds_phys": {
            "x": x_bounds_phys,
            "u": u_bounds_phys,
            "p": p_bounds_phys,
        },
        "var_index": {
            "X_mesh": X_mesh_idx,
            "X_colloc": X_col_idx,
            "U_colloc": U_col_idx,
            "tf": tf_idx,
            "p": p_idx,
        },
        "con_index": {
            "init": init_eq_idx,
            "defect": defect_idx,
            "continuity": cont_idx,
            "boundary": bnd_idx,
            "path": path_idx,
            "state": state_idx,
        },
        "unpack_outputs": ["X_mesh", "X_colloc", "U_colloc", "t_mesh", "t_colloc", "tf"],
    }

    unpack = ca.Function(
        "unpack",
        [wvec],
        [X_mesh_mat, X_col_mat, U_col_mat, t_mesh, t_col, tf_phys],
        ["w"],
        meta["unpack_outputs"],
    )

    return NLP(
        prob=prob,
        lbx=np.array(lbx, dtype=float),
        ubx=np.array(ubx, dtype=float),
        lbg=np.array(lbg, dtype=float),
        ubg=np.array(ubg, dtype=float),
        w0=np.array(w0, dtype=float),
        meta=meta,
        _unpack=unpack,
    )

"""Postprocessing pipeline for direct-collocation NLP solutions."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import DiscreteSolution, NLPLike, PostProcessed
from .common import is_scaled_meta
from .trajectory import CollocationPrimalTrajectory, CollocationDualTrajectory
from .decode import (
    CollocationMeta,
    CollocationDecoded,
    CollocationPrimalGrid,
    CollocationBoundKKT,
    CollocationKKTMultipliers,
    CollocationAdjointGrid,
)


def _np(x: Any, *, squeeze: bool = False) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.squeeze() if squeeze else a


def _slice(w: np.ndarray, idx: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
    if idx is None:
        return None
    a, b = idx
    return w[a:b]
def _unpack_primal(meta: CollocationMeta, named: Dict[str, Any]) -> CollocationPrimalGrid:
    N = meta.N
    K = meta.K
    nx = meta.n_x
    nu = meta.n_u

    x_mesh = _np(named["X_mesh"]).reshape(N + 1, nx)
    x_colloc = _np(named["X_colloc"]).reshape(N, K, nx)
    u_colloc = _np(named["U_colloc"]).reshape(N, K, nu)
    t_mesh = _np(named["t_mesh"], squeeze=True).reshape(N + 1)
    t_colloc = _np(named["t_colloc"], squeeze=True).reshape(N, K)
    tf = float(_np(named["tf"], squeeze=True).reshape(1)[0])

    return CollocationPrimalGrid(
        x_mesh=x_mesh,
        x_colloc=x_colloc,
        u_colloc=u_colloc,
        t_mesh=t_mesh,
        t_colloc=t_colloc,
        tf=tf,
    )


def _unscale_primal(primal_hat: CollocationPrimalGrid, meta: CollocationMeta) -> CollocationPrimalGrid:
    s = meta.scaling
    if s is None:
        return primal_hat

    x_mesh = primal_hat.x_mesh.copy()
    x_colloc = primal_hat.x_colloc.copy()
    u_colloc = primal_hat.u_colloc.copy()

    x_ref = getattr(s, "x_ref", None)
    u_ref = getattr(s, "u_ref", None)

    if x_ref is not None:
        xr = np.asarray(x_ref, float).reshape(1, -1)
        x_mesh = x_mesh * xr
        x_colloc = x_colloc * xr.reshape(1, 1, -1)

    if u_ref is not None:
        ur = np.asarray(u_ref, float).reshape(1, 1, -1)
        u_colloc = u_colloc * ur

    return CollocationPrimalGrid(
        x_mesh=x_mesh,
        x_colloc=x_colloc,
        u_colloc=u_colloc,
        t_mesh=primal_hat.t_mesh,
        t_colloc=primal_hat.t_colloc,
        tf=primal_hat.tf,
    )


def _unpack_kkt_multipliers(meta: CollocationMeta, named: Dict[str, Any], mult_g: np.ndarray) -> CollocationKKTMultipliers:
    cix = meta.con_index()
    N = meta.N
    K = meta.K
    nx = meta.n_x

    t_mesh = _np(named["t_mesh"], squeeze=True).reshape(N + 1)
    t_colloc = _np(named["t_colloc"], squeeze=True).reshape(N, K)
    tf = float(_np(named["tf"], squeeze=True).reshape(1)[0])

    mult_g = _np(mult_g, squeeze=True).reshape(-1)

    if cix.init is not None:
        a, b = cix.init
        nu_init = mult_g[a:b]
    else:
        nu_init = np.zeros((nx,), dtype=float)

    nu_defect = np.zeros((N, K, nx), dtype=float)
    for i in range(N):
        for j in range(K):
            a, b = cix.defect[i][j]
            nu_defect[i, j, :] = mult_g[a:b]

    nu_continuity = np.zeros((N, nx), dtype=float)
    for i in range(N):
        a, b = cix.continuity[i]
        nu_continuity[i, :] = mult_g[a:b]

    nu_boundary = None
    if cix.boundary is not None:
        a, b = cix.boundary
        nu_boundary = mult_g[a:b].copy()

    nu_path = None
    if cix.path is not None:
        nc = None
        for i in range(N):
            for j in range(K):
                ij = cix.path[i][j]
                if ij is not None:
                    a, b = ij
                    nc = b - a
                    break
            if nc is not None:
                break

        if nc is not None and nc > 0:
            nu_path = np.zeros((N, K, nc), dtype=float)
            for i in range(N):
                for j in range(K):
                    ij = cix.path[i][j]
                    if ij is None:
                        continue
                    a, b = ij
                    nu_path[i, j, :] = mult_g[a:b]

    nu_state = None
    if cix.state is not None:
        ns = None
        for i in range(N):
            for j in range(K):
                ij = cix.state[i][j]
                if ij is not None:
                    a, b = ij
                    ns = b - a
                    break
            if ns is not None:
                break

        if ns is not None and ns > 0:
            nu_state = np.zeros((N, K, ns), dtype=float)
            for i in range(N):
                for j in range(K):
                    ij = cix.state[i][j]
                    if ij is None:
                        continue
                    a, b = ij
                    nu_state[i, j, :] = mult_g[a:b]

    return CollocationKKTMultipliers(
        t_mesh=t_mesh,
        t_colloc=t_colloc,
        tf=tf,
        nu_init=nu_init,
        nu_defect=nu_defect,
        nu_continuity=nu_continuity,
        nu_boundary=nu_boundary,
        nu_path=nu_path,
        nu_state=nu_state,
    )


def _unpack_bound_kkt(meta: CollocationMeta, mult_x: np.ndarray) -> CollocationBoundKKT:
    vix = meta.var_index()
    N = meta.N
    K = meta.K
    nx = meta.n_x
    nu = meta.n_u

    mult_x = np.asarray(mult_x, float).reshape(-1)

    signed_x_mesh = np.zeros((N + 1, nx), dtype=float)
    for i, (a, b) in enumerate(vix.X_mesh):
        signed_x_mesh[i, :] = mult_x[a:b]

    signed_x_colloc = np.zeros((N, K, nx), dtype=float)
    for i in range(N):
        for j in range(K):
            a, b = vix.X_colloc[i][j]
            signed_x_colloc[i, j, :] = mult_x[a:b]

    signed_u_colloc = np.zeros((N, K, nu), dtype=float)
    for i in range(N):
        for j in range(K):
            a, b = vix.U_colloc[i][j]
            signed_u_colloc[i, j, :] = mult_x[a:b]

    signed_tf = None
    if vix.tf is not None:
        a, b = vix.tf
        signed_tf = mult_x[a:b].copy()

    signed_p = None
    if vix.p is not None:
        a, b = vix.p
        signed_p = mult_x[a:b].copy()

    return CollocationBoundKKT(
        signed_x_mesh=signed_x_mesh,
        signed_x_colloc=signed_x_colloc,
        signed_u_colloc=signed_u_colloc,
        signed_tf=signed_tf,
        signed_p=signed_p,
    )


def _unscale_bound_kkt(meta: CollocationMeta, bound_hat: CollocationBoundKKT) -> CollocationBoundKKT:
    s = meta.scaling
    if s is None:
        return bound_hat

    signed_x_mesh = bound_hat.signed_x_mesh.copy()
    signed_x_colloc = bound_hat.signed_x_colloc.copy()
    signed_u_colloc = bound_hat.signed_u_colloc.copy()
    signed_tf = None if bound_hat.signed_tf is None else bound_hat.signed_tf.copy()
    signed_p = None if bound_hat.signed_p is None else bound_hat.signed_p.copy()

    x_ref = getattr(s, "x_ref", None)
    u_ref = getattr(s, "u_ref", None)
    t_ref = getattr(s, "t_ref", None)
    p_ref = getattr(s, "p_ref", None)

    if x_ref is not None:
        xr = np.asarray(x_ref, float).reshape(1, -1)
        signed_x_mesh = signed_x_mesh / xr
        signed_x_colloc = signed_x_colloc / xr.reshape(1, 1, -1)

    if u_ref is not None:
        ur = np.asarray(u_ref, float).reshape(1, 1, -1)
        signed_u_colloc = signed_u_colloc / ur

    if signed_tf is not None and t_ref is not None:
        signed_tf = signed_tf / float(t_ref)

    if signed_p is not None and p_ref is not None:
        signed_p = signed_p / np.asarray(p_ref, float).reshape(-1)

    return CollocationBoundKKT(
        signed_x_mesh=signed_x_mesh,
        signed_x_colloc=signed_x_colloc,
        signed_u_colloc=signed_u_colloc,
        signed_tf=signed_tf,
        signed_p=signed_p,
    )


def _unscale_kkt_multipliers(meta: CollocationMeta, kkt_hat: CollocationKKTMultipliers) -> CollocationKKTMultipliers:
    s = meta.scaling
    if s is None:
        return kkt_hat

    x_ref = getattr(s, "x_ref", None)
    if x_ref is None:
        return kkt_hat

    xr = np.asarray(x_ref, float).reshape(-1)

    return CollocationKKTMultipliers(
        t_mesh=kkt_hat.t_mesh,
        t_colloc=kkt_hat.t_colloc,
        tf=kkt_hat.tf,
        nu_init=kkt_hat.nu_init / xr,
        nu_defect=kkt_hat.nu_defect / xr.reshape(1, 1, -1),
        nu_continuity=kkt_hat.nu_continuity / xr.reshape(1, -1),
        nu_boundary=kkt_hat.nu_boundary,
        nu_path=kkt_hat.nu_path,
        nu_state=kkt_hat.nu_state,
    )


def _map_kkt_to_adjoint_grid(
    meta: CollocationMeta,
    kkt: CollocationKKTMultipliers, 
    bound_kkt: Optional[CollocationBoundKKT]=None,
) -> CollocationAdjointGrid:
    N = meta.N
    K = meta.K
    nx = meta.n_x

    nu_init = np.asarray(kkt.nu_init, float).reshape(nx)
    nu_cont = np.asarray(kkt.nu_continuity, float).reshape(N, nx)
    nu_def = np.asarray(kkt.nu_defect, float)

    nu_path = None if kkt.nu_path is None else np.asarray(kkt.nu_path, float)
    nu_state = None if kkt.nu_state is None else np.asarray(kkt.nu_state, float)

    wq = _np(meta.meta.get("quad_weights", []), squeeze=True).reshape(-1)
    if wq.size == 0:
        raise RuntimeError("quad_weights missing in meta; cannot build costate mapping.")
    if wq.size != K:
        raise ValueError(f"quad_weights has length {wq.size}, expected K={K}")

    costate_mesh = np.zeros((N + 1, nx), dtype=float)
    costate_mesh[0, :] = -nu_init
    costate_mesh[1:, :] = nu_cont

    costate_colloc = nu_def / wq[None, :, None]

    path_multiplier_colloc = None
    if nu_path is not None and nu_path.size > 0:
        path_multiplier_colloc = nu_path / wq[None, :, None]

    state_multiplier_colloc = None
    if nu_state is not None and nu_state.size > 0:
        state_multiplier_colloc = np.zeros_like(nu_state, dtype=float)
        for i in range(N):
            for j in range(K):
                for k in range(i):
                    for l in range(K):
                        state_multiplier_colloc[i, j, :] += nu_state[k, l, :]
                for l in range(j):
                    state_multiplier_colloc[i, j, :] += nu_state[i, l, :]


    signed_u_colloc = None

    if bound_kkt is not None:
        signed_u_colloc =  bound_kkt.signed_u_colloc / wq[None, :, None]


    return CollocationAdjointGrid(
        t_mesh=kkt.t_mesh,
        t_colloc=kkt.t_colloc,
        tf=kkt.tf,
        costate_mesh=costate_mesh,
        costate_colloc=costate_colloc,
        path_multiplier_colloc=path_multiplier_colloc,
        state_multiplier_colloc=state_multiplier_colloc,
        signed_u_colloc=signed_u_colloc,
    )


def _extract_residuals(meta: CollocationMeta, gval: np.ndarray) -> Dict[str, np.ndarray]:
    cix = meta.con_index()
    N = meta.N
    K = meta.K
    nx = meta.n_x

    gval = _np(gval, squeeze=True).reshape(-1)
    out: Dict[str, np.ndarray] = {}

    if cix.init is not None:
        a, b = cix.init
        out["r_init"] = gval[a:b].copy()

    r_defect = np.zeros((N, K, nx), dtype=float)
    for i in range(N):
        for j in range(K):
            a, b = cix.defect[i][j]
            r_defect[i, j, :] = gval[a:b]
    out["r_defect"] = r_defect
    out["r_defect_inf"] = float(np.max(np.abs(r_defect)))
    out["r_defect_l2"] = float(np.sqrt(np.mean(r_defect**2)))

    r_continuity = np.zeros((N, nx), dtype=float)
    for i in range(N):
        a, b = cix.continuity[i]
        r_continuity[i, :] = gval[a:b]
    out["r_continuity"] = r_continuity
    out["r_continuity_inf"] = float(np.max(np.abs(r_continuity)))

    if cix.boundary is not None:
        a, b = cix.boundary
        out["r_boundary"] = gval[a:b].copy()

    return out


def postprocess_collocation(ocp: Any, nlp: NLPLike, sol: DiscreteSolution) -> PostProcessed:
    meta = CollocationMeta(nlp.meta or {})
    layout = meta.layout

    named = nlp.unpack(sol.w_opt)

    primal_scaled = _unpack_primal(meta, named)
    primal = _unscale_primal(primal_scaled, meta) if is_scaled_meta(meta) else primal_scaled

    tau = np.asarray(meta.tau, float).reshape(-1)
    s_mesh = np.asarray(layout.s_mesh, float).reshape(-1)

    traj = CollocationPrimalTrajectory(
        primal=primal,
        tau=tau,
        tf=float(primal.tf),
        t0=float(primal.t_mesh[0]),
        s_mesh=s_mesh,
    )

    bound_kkt_scaled = None
    bound_kkt = None
    if sol.mult_x is not None:
        bound_kkt_scaled = _unpack_bound_kkt(meta, sol.mult_x)
        bound_kkt = _unscale_bound_kkt(meta, bound_kkt_scaled) if is_scaled_meta(meta) else bound_kkt_scaled

    kkt_scaled = None
    kkt = None
    adjoint = None
    dual_traj = None
    if sol.mult_g is not None:
        kkt_scaled = _unpack_kkt_multipliers(meta, named, sol.mult_g)
        kkt = _unscale_kkt_multipliers(meta, kkt_scaled) if is_scaled_meta(meta) else kkt_scaled
        adjoint = _map_kkt_to_adjoint_grid(meta, kkt, bound_kkt)

        dual_traj = CollocationDualTrajectory(
            dual=adjoint,
            tau=tau,
            tf=float(adjoint.tf),
            t0=float(adjoint.t_mesh[0]),
            s_mesh=s_mesh,
        )

    decoded = CollocationDecoded(
        layout=layout,
        primal=primal,
        primal_scaled=primal_scaled if is_scaled_meta(meta) else None,
        kkt=kkt,
        kkt_scaled=kkt_scaled if is_scaled_meta(meta) else None,
        bound_kkt=bound_kkt,
        bound_kkt_scaled=bound_kkt_scaled if is_scaled_meta(meta) else None,
        adjoint=adjoint,
    )

    diag: Dict[str, Any] = {
        "tf": float(primal.tf),
        "N": meta.N,
        "degree": meta.K,
        "tau": tau,
        "s_mesh": s_mesh,
        "n_x": meta.n_x,
        "n_u": meta.n_u,
        "space": meta.space,
        "bounds_phys": meta.bounds_phys,
        "bounds_ocp": meta.bounds_ocp,
    }

    try:
        gval = nlp.eval_g(sol.w_opt)
        diag["residuals"] = _extract_residuals(meta, gval)
    except Exception as e:
        diag["residuals_error"] = repr(e)

    return PostProcessed(
        traj=traj,
        dual_traj=dual_traj,
        decoded=decoded,
        diag=diag,
    )

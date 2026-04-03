"""Serialization helpers for collocation solutions.

This module converts between in-memory postprocessed collocation structures and
portable :class:`~casadi_control.discretization.base.SolutionArtifact` payloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json
import numpy as np

from ..base import DiscreteSolution, PostProcessed, SolutionArtifact
from .trajectory import CollocationPrimalTrajectory, CollocationDualTrajectory
from .decode import (
    CollocationLayout,
    CollocationDecoded,
    CollocationPrimalGrid,
    CollocationKKTMultipliers,
    CollocationBoundKKT,
    CollocationAdjointGrid,
)


# =============================================================================
# JSON helpers
# =============================================================================

def _jsonify(obj: Any) -> Any:
    """Convert numpy-heavy nested structures to JSON-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _pack_json(d: Dict[str, Any]) -> str:
    """Serialize a metadata dictionary to canonical JSON."""
    return json.dumps(_jsonify(d), sort_keys=True)


def _unpack_json(s: str) -> Dict[str, Any]:
    """Deserialize and validate metadata JSON payload."""
    out = json.loads(s)
    if not isinstance(out, dict):
        raise TypeError("artifact meta json must decode to a dict")
    return out


# =============================================================================
# Generic NPZ helpers
# =============================================================================

def save_npz(path: str | Path, art: SolutionArtifact) -> None:
    """Persist a :class:`SolutionArtifact` to compressed NPZ format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "__discretization__": np.array(art.discretization),
        "__meta_json__": np.array(_pack_json(art.meta)),
    }

    for k, v in art.arrays.items():
        payload[str(k)] = np.asarray(v)

    np.savez_compressed(str(path), **payload)


def load_npz(path: str | Path) -> SolutionArtifact:
    """Load a :class:`SolutionArtifact` from compressed NPZ format."""
    d = np.load(str(path), allow_pickle=False)

    discretization = str(np.asarray(d["__discretization__"]).item())
    meta = _unpack_json(str(np.asarray(d["__meta_json__"]).item()))

    arrays: Dict[str, np.ndarray] = {}
    for k in d.files:
        if k.startswith("__"):
            continue
        arrays[k] = np.asarray(d[k])

    return SolutionArtifact(
        discretization=discretization,
        arrays=arrays,
        meta=meta,
    )


# =============================================================================
# Small helpers
# =============================================================================

def _opt_array(arrays: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    """Return None for missing/empty arrays and ndarray otherwise."""
    a = arrays.get(key, None)
    if a is None:
        return None
    a = np.asarray(a)
    return None if a.size == 0 else a


def _req_array(arrays: Dict[str, np.ndarray], key: str) -> np.ndarray:
    """Return a required array or raise a helpful error."""
    if key not in arrays:
        raise KeyError(f"artifact is missing required array {key!r}")
    return np.asarray(arrays[key])


def _empty_if_none(a: Optional[np.ndarray]) -> np.ndarray:
    """Encode optional arrays as empty arrays in the artifact."""
    if a is None:
        return np.asarray([], dtype=float)
    return np.asarray(a, dtype=float)


def _uniform_s_mesh_from_t_mesh(t_mesh: np.ndarray, *, t0: float, tf: float) -> np.ndarray:
    """
    Backward fallback if old artifacts do not have s_mesh:
    infer s_mesh from t_mesh via (t_mesh - t0)/tf, and if that looks bad use uniform.
    """
    t_mesh = np.asarray(t_mesh, float).reshape(-1)
    N = t_mesh.size - 1
    if tf <= 0.0 or N <= 0:
        return np.linspace(0.0, 1.0, max(N + 1, 2), dtype=float)

    s = (t_mesh - float(t0)) / float(tf)
    if s.size == N + 1 and np.all(np.diff(s) > 0.0):
        s = s.copy()
        s[0] = 0.0
        s[-1] = 1.0
        return s

    return np.linspace(0.0, 1.0, N + 1, dtype=float)


def _layout_to_meta_dict(layout: CollocationLayout) -> Dict[str, Any]:
    """Serialize layout into JSON metadata."""
    return {
        "N": int(layout.N),
        "K": int(layout.K),
        "tau": np.asarray(layout.tau, float).tolist(),
        "s_mesh": np.asarray(layout.s_mesh, float).tolist(),
        "t0": float(layout.t0),
    }


def _layout_from_meta_dict(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> CollocationLayout:
    """Recover layout from artifact metadata, with backward-compatible fallbacks."""
    layout_meta = meta.get("layout", None)
    if isinstance(layout_meta, dict):
        return CollocationLayout(
            N=int(layout_meta["N"]),
            K=int(layout_meta["K"]),
            tau=np.asarray(layout_meta["tau"], float),
            s_mesh=np.asarray(layout_meta["s_mesh"], float),
            t0=float(layout_meta.get("t0", 0.0)),
        )

    tf = float(np.asarray(arrays["tf"]).reshape(-1)[0])
    tau = np.asarray(arrays["tau"], float).reshape(-1)
    t_mesh = np.asarray(arrays["t_mesh"], float).reshape(-1)

    s_mesh = _opt_array(arrays, "s_mesh")
    if s_mesh is None or s_mesh.size == 0:
        t0 = float(t_mesh[0]) if t_mesh.size > 0 else 0.0
        s_mesh = _uniform_s_mesh_from_t_mesh(t_mesh, t0=t0, tf=tf)
    else:
        s_mesh = np.asarray(s_mesh, float).reshape(-1)

    return CollocationLayout(
        N=int(t_mesh.size - 1),
        K=int(tau.size - 1),
        tau=tau,
        s_mesh=s_mesh,
        t0=float(t_mesh[0]) if t_mesh.size > 0 else 0.0,
    )


# =============================================================================
# Packing helpers
# =============================================================================

def _pack_primal(prefix: str, primal: Optional[CollocationPrimalGrid], arrays: Dict[str, np.ndarray]) -> None:
    """Pack a primal grid under the given prefix."""
    if primal is None:
        arrays[f"{prefix}_present"] = np.asarray([0], dtype=int)
        return

    arrays[f"{prefix}_present"] = np.asarray([1], dtype=int)
    arrays[f"{prefix}_x_mesh"] = np.asarray(primal.x_mesh, float)
    arrays[f"{prefix}_x_colloc"] = np.asarray(primal.x_colloc, float)
    arrays[f"{prefix}_u_colloc"] = np.asarray(primal.u_colloc, float)
    arrays[f"{prefix}_t_mesh"] = np.asarray(primal.t_mesh, float).reshape(-1)
    arrays[f"{prefix}_t_colloc"] = np.asarray(primal.t_colloc, float)
    arrays[f"{prefix}_tf"] = np.asarray([float(primal.tf)], float)


def _unpack_primal(prefix: str, arrays: Dict[str, np.ndarray]) -> Optional[CollocationPrimalGrid]:
    """Unpack a primal grid from the given prefix."""
    present = _opt_array(arrays, f"{prefix}_present")
    if present is None or int(np.asarray(present).reshape(-1)[0]) == 0:
        return None

    return CollocationPrimalGrid(
        x_mesh=np.asarray(_req_array(arrays, f"{prefix}_x_mesh"), float),
        x_colloc=np.asarray(_req_array(arrays, f"{prefix}_x_colloc"), float),
        u_colloc=np.asarray(_req_array(arrays, f"{prefix}_u_colloc"), float),
        t_mesh=np.asarray(_req_array(arrays, f"{prefix}_t_mesh"), float).reshape(-1),
        t_colloc=np.asarray(_req_array(arrays, f"{prefix}_t_colloc"), float),
        tf=float(np.asarray(_req_array(arrays, f"{prefix}_tf")).reshape(-1)[0]),
    )


def _pack_kkt(prefix: str, kkt: Optional[CollocationKKTMultipliers], arrays: Dict[str, np.ndarray]) -> None:
    """Pack decoded NLP constraint multipliers."""
    if kkt is None:
        arrays[f"{prefix}_present"] = np.asarray([0], dtype=int)
        return

    arrays[f"{prefix}_present"] = np.asarray([1], dtype=int)
    arrays[f"{prefix}_t_mesh"] = np.asarray(kkt.t_mesh, float).reshape(-1)
    arrays[f"{prefix}_t_colloc"] = np.asarray(kkt.t_colloc, float)
    arrays[f"{prefix}_tf"] = np.asarray([float(kkt.tf)], float)

    arrays[f"{prefix}_nu_init"] = np.asarray(kkt.nu_init, float)
    arrays[f"{prefix}_nu_defect"] = np.asarray(kkt.nu_defect, float)
    arrays[f"{prefix}_nu_continuity"] = np.asarray(kkt.nu_continuity, float)
    arrays[f"{prefix}_nu_boundary"] = _empty_if_none(kkt.nu_boundary)
    arrays[f"{prefix}_nu_path"] = _empty_if_none(kkt.nu_path)
    arrays[f"{prefix}_nu_state"] = _empty_if_none(kkt.nu_state)


def _unpack_kkt(prefix: str, arrays: Dict[str, np.ndarray]) -> Optional[CollocationKKTMultipliers]:
    """Unpack decoded NLP constraint multipliers."""
    present = _opt_array(arrays, f"{prefix}_present")
    if present is None or int(np.asarray(present).reshape(-1)[0]) == 0:
        return None

    nu_boundary = _opt_array(arrays, f"{prefix}_nu_boundary")
    nu_path = _opt_array(arrays, f"{prefix}_nu_path")
    nu_state = _opt_array(arrays, f"{prefix}_nu_state")

    return CollocationKKTMultipliers(
        t_mesh=np.asarray(_req_array(arrays, f"{prefix}_t_mesh"), float).reshape(-1),
        t_colloc=np.asarray(_req_array(arrays, f"{prefix}_t_colloc"), float),
        tf=float(np.asarray(_req_array(arrays, f"{prefix}_tf")).reshape(-1)[0]),
        nu_init=np.asarray(_req_array(arrays, f"{prefix}_nu_init"), float),
        nu_defect=np.asarray(_req_array(arrays, f"{prefix}_nu_defect"), float),
        nu_continuity=np.asarray(_req_array(arrays, f"{prefix}_nu_continuity"), float),
        nu_boundary=None if nu_boundary is None else np.asarray(nu_boundary, float),
        nu_path=None if nu_path is None else np.asarray(nu_path, float),
        nu_state=None if nu_state is None else np.asarray(nu_state, float),
    )


def _pack_bound_kkt(prefix: str, bound_kkt: Optional[CollocationBoundKKT], arrays: Dict[str, np.ndarray]) -> None:
    """Pack decoded variable-bound multipliers."""
    if bound_kkt is None:
        arrays[f"{prefix}_present"] = np.asarray([0], dtype=int)
        return

    arrays[f"{prefix}_present"] = np.asarray([1], dtype=int)
    arrays[f"{prefix}_signed_x_mesh"] = np.asarray(bound_kkt.signed_x_mesh, float)
    arrays[f"{prefix}_signed_x_colloc"] = np.asarray(bound_kkt.signed_x_colloc, float)
    arrays[f"{prefix}_signed_u_colloc"] = np.asarray(bound_kkt.signed_u_colloc, float)
    arrays[f"{prefix}_signed_tf"] = _empty_if_none(bound_kkt.signed_tf)
    arrays[f"{prefix}_signed_p"] = _empty_if_none(bound_kkt.signed_p)


def _unpack_bound_kkt(prefix: str, arrays: Dict[str, np.ndarray]) -> Optional[CollocationBoundKKT]:
    """Unpack decoded variable-bound multipliers."""
    present = _opt_array(arrays, f"{prefix}_present")
    if present is None or int(np.asarray(present).reshape(-1)[0]) == 0:
        return None

    signed_tf = _opt_array(arrays, f"{prefix}_signed_tf")
    signed_p = _opt_array(arrays, f"{prefix}_signed_p")

    return CollocationBoundKKT(
        signed_x_mesh=np.asarray(_req_array(arrays, f"{prefix}_signed_x_mesh"), float),
        signed_x_colloc=np.asarray(_req_array(arrays, f"{prefix}_signed_x_colloc"), float),
        signed_u_colloc=np.asarray(_req_array(arrays, f"{prefix}_signed_u_colloc"), float),
        signed_tf=None if signed_tf is None else np.asarray(signed_tf, float),
        signed_p=None if signed_p is None else np.asarray(signed_p, float),
    )


def _pack_adjoint(prefix: str, adjoint: Optional[CollocationAdjointGrid], arrays: Dict[str, np.ndarray]) -> None:
    """Pack continuous dual quantities."""
    if adjoint is None:
        arrays[f"{prefix}_present"] = np.asarray([0], dtype=int)
        return

    arrays[f"{prefix}_present"] = np.asarray([1], dtype=int)
    arrays[f"{prefix}_t_mesh"] = np.asarray(adjoint.t_mesh, float).reshape(-1)
    arrays[f"{prefix}_t_colloc"] = np.asarray(adjoint.t_colloc, float)
    arrays[f"{prefix}_tf"] = np.asarray([float(adjoint.tf)], float)

    arrays[f"{prefix}_costate_mesh"] = np.asarray(adjoint.costate_mesh, float)
    arrays[f"{prefix}_costate_colloc"] = np.asarray(adjoint.costate_colloc, float)
    arrays[f"{prefix}_path_multiplier_colloc"] = _empty_if_none(adjoint.path_multiplier_colloc)
    arrays[f"{prefix}_state_multiplier_colloc"] = _empty_if_none(adjoint.state_multiplier_colloc)


def _unpack_adjoint(prefix: str, arrays: Dict[str, np.ndarray]) -> Optional[CollocationAdjointGrid]:
    """Unpack continuous dual quantities."""
    present = _opt_array(arrays, f"{prefix}_present")
    if present is None or int(np.asarray(present).reshape(-1)[0]) == 0:
        return None

    path_multiplier_colloc = _opt_array(arrays, f"{prefix}_path_multiplier_colloc")
    state_multiplier_colloc = _opt_array(arrays, f"{prefix}_state_multiplier_colloc")

    return CollocationAdjointGrid(
        t_mesh=np.asarray(_req_array(arrays, f"{prefix}_t_mesh"), float).reshape(-1),
        t_colloc=np.asarray(_req_array(arrays, f"{prefix}_t_colloc"), float),
        tf=float(np.asarray(_req_array(arrays, f"{prefix}_tf")).reshape(-1)[0]),
        costate_mesh=np.asarray(_req_array(arrays, f"{prefix}_costate_mesh"), float),
        costate_colloc=np.asarray(_req_array(arrays, f"{prefix}_costate_colloc"), float),
        path_multiplier_colloc=None if path_multiplier_colloc is None else np.asarray(path_multiplier_colloc, float),
        state_multiplier_colloc=None if state_multiplier_colloc is None else np.asarray(state_multiplier_colloc, float),
    )


# =============================================================================
# Collocation-specific conversion
# =============================================================================

def collocation_to_artifact(sol: DiscreteSolution, pp: PostProcessed) -> SolutionArtifact:
    """
    Convert collocation postprocessed output to a scheme-agnostic artifact.

    Data contract
    -------------
    - decoded primal arrays are physical
    - decoded *_scaled arrays are in solver/scaled coordinates
    - decoded adjoint arrays are the interpreted continuous dual objects
    - decoded KKT arrays are the discrete NLP duals
    - decoded bound-KKT arrays are variable-bound multipliers
    - raw solver vectors are also stored for debugging / re-decoding
    """
    decoded = pp.decoded
    if not isinstance(decoded, CollocationDecoded):
        raise TypeError("pp.decoded must be a CollocationDecoded")

    arrays: Dict[str, np.ndarray] = {}

    _pack_primal("primal", decoded.primal, arrays)
    _pack_primal("primal_scaled", decoded.primal_scaled, arrays)

    _pack_kkt("kkt", decoded.kkt, arrays)
    _pack_kkt("kkt_scaled", decoded.kkt_scaled, arrays)

    _pack_bound_kkt("bound_kkt", decoded.bound_kkt, arrays)
    _pack_bound_kkt("bound_kkt_scaled", decoded.bound_kkt_scaled, arrays)

    _pack_adjoint("adjoint", decoded.adjoint, arrays)

    arrays["tau"] = np.asarray(decoded.layout.tau, float).reshape(-1)
    arrays["s_mesh"] = np.asarray(decoded.layout.s_mesh, float).reshape(-1)

    arrays["w_opt"] = np.asarray(sol.w_opt, float).reshape(-1)
    arrays["mult_x_raw"] = (
        np.asarray(sol.mult_x, float).reshape(-1)
        if sol.mult_x is not None
        else np.asarray([], dtype=float)
    )
    arrays["mult_g_raw"] = (
        np.asarray(sol.mult_g, float).reshape(-1)
        if sol.mult_g is not None
        else np.asarray([], dtype=float)
    )

    meta = {
        "status": str(sol.status),
        "f_opt": float(sol.f_opt),
        "diag": pp.diag,
        "stats": sol.stats or {},
        "layout": _layout_to_meta_dict(decoded.layout),
    }

    return SolutionArtifact(
        discretization="direct_collocation",
        arrays=arrays,
        meta=meta,
    )


def collocation_from_artifact(art: SolutionArtifact) -> PostProcessed:
    """
    Convert a collocation artifact into a plot-ready PostProcessed object.
    """
    if art.discretization != "direct_collocation":
        raise ValueError(f"Expected discretization='direct_collocation', got {art.discretization!r}")

    arrays = art.arrays
    layout = _layout_from_meta_dict(art.meta, arrays)

    primal = _unpack_primal("primal", arrays)
    if primal is None:
        raise ValueError("artifact is missing required decoded primal data")

    primal_scaled = _unpack_primal("primal_scaled", arrays)
    kkt = _unpack_kkt("kkt", arrays)
    kkt_scaled = _unpack_kkt("kkt_scaled", arrays)
    bound_kkt = _unpack_bound_kkt("bound_kkt", arrays)
    bound_kkt_scaled = _unpack_bound_kkt("bound_kkt_scaled", arrays)
    adjoint = _unpack_adjoint("adjoint", arrays)

    traj = CollocationPrimalTrajectory(
        primal=primal,
        tau=np.asarray(layout.tau, float).reshape(-1),
        tf=float(primal.tf),
        t0=float(layout.t0),
        s_mesh=np.asarray(layout.s_mesh, float).reshape(-1),
    )

    dual_traj = None
    if adjoint is not None:
        dual_traj = CollocationDualTrajectory(
            dual=adjoint,
            tau=np.asarray(layout.tau, float).reshape(-1),
            tf=float(adjoint.tf),
            t0=float(layout.t0),
            s_mesh=np.asarray(layout.s_mesh, float).reshape(-1),
        )

    decoded = CollocationDecoded(
        layout=layout,
        primal=primal,
        primal_scaled=primal_scaled,
        kkt=kkt,
        kkt_scaled=kkt_scaled,
        bound_kkt=bound_kkt,
        bound_kkt_scaled=bound_kkt_scaled,
        adjoint=adjoint,
    )

    diag = dict(art.meta.get("diag", {}))
    diag["status"] = art.meta.get("status", None)
    diag["f_opt"] = art.meta.get("f_opt", None)
    diag.setdefault("tau", np.asarray(layout.tau, float).reshape(-1))
    diag.setdefault("s_mesh", np.asarray(layout.s_mesh, float).reshape(-1))

    raw_mult_x = _opt_array(arrays, "mult_x_raw")
    raw_mult_g = _opt_array(arrays, "mult_g_raw")
    if raw_mult_x is not None:
        diag["mult_x_raw"] = np.asarray(raw_mult_x, float).reshape(-1)
    if raw_mult_g is not None:
        diag["mult_g_raw"] = np.asarray(raw_mult_g, float).reshape(-1)

    return PostProcessed(
        traj=traj,
        dual_traj=dual_traj,
        decoded=decoded,
        diag=diag,
    )

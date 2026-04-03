"""Typed decoded collocation data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeAlias

import numpy as np


IndexSpan: TypeAlias = tuple[int, int]
IndexGrid: TypeAlias = list[list[IndexSpan]]
OptionalIndexGrid: TypeAlias = list[list[Optional[IndexSpan]]]


@dataclass(frozen=True)
class CollocationVarIndex:
    """Typed variable-block indices for a collocation NLP."""

    X_mesh: list[IndexSpan]
    X_colloc: IndexGrid
    U_colloc: IndexGrid
    tf: Optional[IndexSpan] = None
    p: Optional[IndexSpan] = None


@dataclass(frozen=True)
class CollocationConIndex:
    """Typed constraint-block indices for a collocation NLP."""

    init: Optional[IndexSpan]
    defect: IndexGrid
    continuity: list[IndexSpan]
    boundary: Optional[IndexSpan] = None
    path: OptionalIndexGrid | None = None
    state: OptionalIndexGrid | None = None


# =============================================================================
# Layout / metadata
# =============================================================================

@dataclass(frozen=True)
class CollocationLayout:
    """
    Structural description of a direct-collocation discretization.

    Conventions
    -----------
    - tau has shape (K+1,) with tau[0] = 0 and collocation nodes tau[1:].
    - s_mesh has shape (N+1,) and stores the normalized mesh in [0, 1].
    """
    N: int
    K: int
    tau: np.ndarray
    s_mesh: np.ndarray
    t0: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "N", int(self.N))
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "tau", np.asarray(self.tau, float).reshape(-1))
        object.__setattr__(self, "s_mesh", np.asarray(self.s_mesh, float).reshape(-1))
        object.__setattr__(self, "t0", float(self.t0))

        if self.tau.size != self.K + 1:
            raise ValueError(f"tau must have size K+1={self.K+1}, got {self.tau.size}")
        if abs(float(self.tau[0])) > 1e-14:
            raise ValueError("tau[0] must equal 0.0")

        if self.s_mesh.size != self.N + 1:
            raise ValueError(f"s_mesh must have size N+1={self.N+1}, got {self.s_mesh.size}")
        if not np.all(np.diff(self.s_mesh) > 0.0):
            raise ValueError("s_mesh must be strictly increasing")
        if abs(float(self.s_mesh[0])) > 1e-12 or abs(float(self.s_mesh[-1]) - 1.0) > 1e-12:
            raise ValueError("s_mesh must start at 0 and end at 1")

    def times_numpy(self, *, tf: float) -> Dict[str, np.ndarray]:
        """
        Return time grids in the layout's time coordinates.
        """
        tf = float(tf)

        t_mesh = self.t0 + tf * self.s_mesh

        t_colloc = np.zeros((self.N, self.K), dtype=float)
        for i in range(self.N):
            ds_i = self.s_mesh[i + 1] - self.s_mesh[i]
            for j in range(self.K):
                t_colloc[i, j] = self.t0 + tf * (self.s_mesh[i] + ds_i * float(self.tau[j + 1]))

        return {
            "t_mesh": t_mesh,
            "t_colloc": t_colloc,
            "t_colloc_flat": t_colloc.reshape(-1),
        }


@dataclass(frozen=True)
class CollocationMeta:
    """
    Typed accessor wrapper around an NLP ``meta`` dictionary.
    """
    meta: Dict[str, Any]

    @property
    def layout(self) -> CollocationLayout:
        lay = self.meta["layout"]
        if not isinstance(lay, CollocationLayout):
            raise TypeError("meta['layout'] must be a CollocationLayout")
        return lay

    @property
    def N(self) -> int:
        return int(self.meta["N"])

    @property
    def K(self) -> int:
        return int(self.meta["degree"])

    @property
    def n_x(self) -> int:
        return int(self.meta["n_x"])

    @property
    def n_u(self) -> int:
        return int(self.meta["n_u"])

    @property
    def tau(self) -> np.ndarray:
        return np.asarray(self.meta["tau"], float).reshape(-1)

    @property
    def space(self) -> str:
        return str(self.meta["space"])

    @property
    def t_ref(self) -> Optional[float]:
        val = self.meta.get("t_ref", None)
        return None if val is None else float(val)

    @property
    def t0_phys(self) -> float:
        return float(self.meta["t0_phys"])

    @property
    def scaling(self) -> Any:
        return self.meta.get("scaling", None)

    @property
    def bounds_ocp(self) -> Dict[str, Any]:
        b = self.meta["bounds_ocp"]
        if not isinstance(b, dict):
            raise TypeError("meta['bounds_ocp'] must be a dict")
        return b

    @property
    def bounds_phys(self) -> Dict[str, Any]:
        b = self.meta["bounds_phys"]
        if not isinstance(b, dict):
            raise TypeError("meta['bounds_phys'] must be a dict")
        return b

    def var_index(self) -> CollocationVarIndex:
        vix = self.meta["var_index"]
        if not isinstance(vix, CollocationVarIndex):
            raise TypeError("meta['var_index'] must be a CollocationVarIndex")
        return vix

    def con_index(self) -> CollocationConIndex:
        cix = self.meta["con_index"]
        if not isinstance(cix, CollocationConIndex):
            raise TypeError("meta['con_index'] must be a CollocationConIndex")
        return cix


# =============================================================================
# Decoded primal grid
# =============================================================================

@dataclass(frozen=True)
class CollocationPrimalGrid:
    """Decoded primal values on mesh and collocation nodes.

    This is the main grid-level primal payload stored in
    ``PostProcessed.decoded`` for direct collocation. All arrays here are
    already unpacked from the flat NLP decision vector.

    Attributes
    ----------
    x_mesh : ndarray
        State values at mesh nodes, shape ``(N+1, nx)``.
    x_colloc : ndarray
        State values at collocation nodes, shape ``(N, K, nx)``.
    u_colloc : ndarray
        Control values at collocation nodes, shape ``(N, K, nu)``.
    t_mesh : ndarray
        Physical-time mesh nodes, shape ``(N+1,)``.
    t_colloc : ndarray
        Physical-time collocation nodes, shape ``(N, K)``.
    tf : float
        Physical final time.

    Notes
    -----
    Use this object when you want the actual collocation-node data used by the
    transcription, for example when plotting node values, checking mesh
    refinement behavior, or building custom residual diagnostics.
    """
    x_mesh: np.ndarray         # (N+1, nx)
    x_colloc: np.ndarray       # (N, K, nx)
    u_colloc: np.ndarray       # (N, K, nu)
    t_mesh: np.ndarray         # (N+1,)
    t_colloc: np.ndarray       # (N, K)
    tf: float

    @property
    def x_colloc_flat(self) -> np.ndarray:
        N, K, nx = self.x_colloc.shape
        return self.x_colloc.reshape(N * K, nx)

    @property
    def u_colloc_flat(self) -> np.ndarray:
        N, K, nu = self.u_colloc.shape
        return self.u_colloc.reshape(N * K, nu)

    @property
    def t_colloc_flat(self) -> np.ndarray:
        return self.t_colloc.reshape(-1)

    @property
    def t_nodes(self) -> np.ndarray:
        N, K = self.t_colloc.shape
        out = np.empty(N * (K + 1) + 1, dtype=float)

        p = 0
        for i in range(N):
            out[p] = self.t_mesh[i]
            p += 1
            out[p:p + K] = self.t_colloc[i, :]
            p += K
        out[p] = self.t_mesh[-1]
        return out

    @property
    def x_nodes(self) -> np.ndarray:
        N, K, nx = self.x_colloc.shape
        out = np.empty((N * (K + 1) + 1, nx), dtype=float)

        p = 0
        for i in range(N):
            out[p, :] = self.x_mesh[i, :]
            p += 1
            out[p:p + K, :] = self.x_colloc[i, :, :]
            p += K
        out[p, :] = self.x_mesh[-1, :]
        return out


# =============================================================================
# Decoded NLP multipliers
# =============================================================================

@dataclass(frozen=True)
class CollocationKKTMultipliers:
    """Decoded NLP constraint multipliers.

    These correspond to constraints as encoded in the NLP, not yet to continuous
    costates or continuous multiplier fields.

    In other words, this object reflects the solver's dual variables on the
    discrete transcription grid. If you want interpreted costates and
    multiplier trajectories, use :class:`CollocationAdjointGrid` instead.
    """
    t_mesh: np.ndarray
    t_colloc: np.ndarray
    tf: float

    nu_init: np.ndarray
    nu_defect: np.ndarray
    nu_continuity: np.ndarray

    nu_boundary: Optional[np.ndarray] = None
    nu_path: Optional[np.ndarray] = None
    nu_state: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CollocationBoundKKT:
    """
    Decoded variable-bound multipliers on decision-variable blocks.

    The stored arrays preserve the signed solver convention in `signed_*`.
    The `lower_*` / `upper_*` properties provide a convenient decomposition using

        lower = max(signed, 0)
        upper = max(-signed, 0)

    which matches the usual IPOPT/CasADi convention for lam_x. Verify once against
    your solver adapter if you want to rely on lower/upper semantics.
    """
    signed_x_mesh: np.ndarray
    signed_x_colloc: np.ndarray
    signed_u_colloc: np.ndarray

    signed_tf: Optional[np.ndarray] = None
    signed_p: Optional[np.ndarray] = None

    @staticmethod
    def _lower(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if a is None else np.maximum(a, 0.0)

    @staticmethod
    def _upper(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if a is None else np.maximum(-a, 0.0)

    @property
    def lower_x_mesh(self) -> np.ndarray:
        return self._lower(self.signed_x_mesh)

    @property
    def upper_x_mesh(self) -> np.ndarray:
        return self._upper(self.signed_x_mesh)

    @property
    def lower_x_colloc(self) -> np.ndarray:
        return self._lower(self.signed_x_colloc)

    @property
    def upper_x_colloc(self) -> np.ndarray:
        return self._upper(self.signed_x_colloc)

    @property
    def lower_u_colloc(self) -> np.ndarray:
        return self._lower(self.signed_u_colloc)

    @property
    def upper_u_colloc(self) -> np.ndarray:
        return self._upper(self.signed_u_colloc)

    @property
    def lower_tf(self) -> Optional[np.ndarray]:
        return self._lower(self.signed_tf)

    @property
    def upper_tf(self) -> Optional[np.ndarray]:
        return self._upper(self.signed_tf)

    @property
    def lower_p(self) -> Optional[np.ndarray]:
        return self._lower(self.signed_p)

    @property
    def upper_p(self) -> Optional[np.ndarray]:
        return self._upper(self.signed_p)


# =============================================================================
# Continuous dual grid
# =============================================================================

@dataclass(frozen=True)
class CollocationAdjointGrid:
    """Continuous dual quantities represented on mesh and collocation nodes.

    This object is the dual counterpart of :class:`CollocationPrimalGrid`.
    Unlike :class:`CollocationKKTMultipliers`, its arrays are intended to be
    interpreted as costate-like and multiplier-like quantities aligned with the
    postprocessed primal trajectory.
    """
    t_mesh: np.ndarray
    t_colloc: np.ndarray
    tf: float

    costate_mesh: np.ndarray
    costate_colloc: np.ndarray

    path_multiplier_colloc:  Optional[np.ndarray] = None
    state_multiplier_colloc: Optional[np.ndarray] = None

    signed_x_mesh:   Optional[np.ndarray] = None
    signed_x_colloc: Optional[np.ndarray] = None
    signed_u_colloc: Optional[np.ndarray] = None

    @property
    def t_colloc_flat(self) -> np.ndarray:
        return self.t_colloc.reshape(-1)

    @property
    def costate_colloc_flat(self) -> np.ndarray:
        N, K, nx = self.costate_colloc.shape
        return self.costate_colloc.reshape(N * K, nx)

    @property
    def t_nodes(self) -> np.ndarray:
        N, K = self.t_colloc.shape
        out = np.empty(N * (K + 1) + 1, dtype=float)

        p = 0
        for i in range(N):
            out[p] = self.t_mesh[i]
            p += 1
            out[p:p + K] = self.t_colloc[i, :]
            p += K
        out[p] = self.t_mesh[-1]
        return out

    @property
    def costate_nodes(self) -> np.ndarray:
        N, K, nx = self.costate_colloc.shape
        out = np.empty((N * (K + 1) + 1, nx), dtype=float)

        p = 0
        for i in range(N):
            out[p, :] = self.costate_mesh[i, :]
            p += 1
            out[p:p + K, :] = self.costate_colloc[i, :, :]
            p += K
        out[p, :] = self.costate_mesh[-1, :]
        return out

    @staticmethod
    def _lower(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if a is None else np.maximum(-a, 0.0)

    @staticmethod
    def _upper(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if a is None else np.maximum(a, 0.0)

    @property
    def lower_x_mesh(self) -> np.ndarray:
        return self._lower(self.signed_x_mesh)

    @property
    def upper_x_mesh(self) -> np.ndarray:
        return self._upper(self.signed_x_mesh)

    @property
    def lower_x_colloc(self) -> np.ndarray:
        return self._lower(self.signed_x_colloc)

    @property
    def upper_x_colloc(self) -> np.ndarray:
        return self._upper(self.signed_x_colloc)

    @property
    def lower_u_colloc(self) -> np.ndarray:
        return self._lower(self.signed_u_colloc)

    @property
    def upper_u_colloc(self) -> np.ndarray:
        return self._upper(self.signed_u_colloc)

# =============================================================================
# Full decoded payload
# =============================================================================

@dataclass(frozen=True)
class CollocationDecoded:
    """Full decoded collocation postprocessing payload.

    This is the concrete type stored in ``pp.decoded`` after
    :meth:`casadi_control.discretization.collocation.DirectCollocation.postprocess`.

    Attributes
    ----------
    layout : CollocationLayout
        Structural metadata for the discretization: number of intervals,
        collocation degree, normalized mesh, collocation nodes, and initial
        time.
    primal : CollocationPrimalGrid
        Physical-time primal arrays unpacked from the solution.
    primal_scaled : CollocationPrimalGrid, optional
        Solver/scaled-coordinate version of the primal arrays when scaling is
        active.
    kkt, kkt_scaled : CollocationKKTMultipliers, optional
        Decoded NLP constraint multipliers in physical and solver coordinates.
    bound_kkt, bound_kkt_scaled : CollocationBoundKKT, optional
        Decoded decision-variable bound multipliers in physical and solver
        coordinates.
    adjoint : CollocationAdjointGrid, optional
        Interpreted continuous dual quantities reconstructed from the NLP
        multipliers.

    Notes
    -----
    ``CollocationDecoded`` is useful when you need the collocation solution in
    the coordinates and block structure of the transcription itself. Common
    examples include:

    - plotting mesh and collocation node values directly
    - comparing physical and scaled variables
    - inspecting raw NLP multipliers before or after dual interpretation
    - exporting structured arrays for custom reports or regression tests
    """
    layout: CollocationLayout

    primal: CollocationPrimalGrid
    primal_scaled: Optional[CollocationPrimalGrid] = None

    kkt: Optional[CollocationKKTMultipliers] = None
    kkt_scaled: Optional[CollocationKKTMultipliers] = None

    bound_kkt: Optional[CollocationBoundKKT] = None
    bound_kkt_scaled: Optional[CollocationBoundKKT] = None

    adjoint: Optional[CollocationAdjointGrid] = None

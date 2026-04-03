"""Typed decoded collocation data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


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
    def tau(self) -> np.ndarray:
        return np.asarray(self.meta["tau"], float).reshape(-1)

    @property
    def space(self) -> str:
        return str(self.meta["space"])

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

    def var_index(self) -> Dict[str, Any]:
        vix = self.meta["var_index"]
        if not isinstance(vix, dict):
            raise TypeError("meta['var_index'] must be a dict")
        return vix

    def con_index(self) -> Dict[str, Any]:
        cix = self.meta["con_index"]
        if not isinstance(cix, dict):
            raise TypeError("meta['con_index'] must be a dict")
        return cix


# =============================================================================
# Decoded primal grid
# =============================================================================

@dataclass(frozen=True)
class CollocationPrimalGrid:
    """
    Decoded primal values on mesh and collocation nodes.
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
    """
    Decoded NLP constraint multipliers.

    These correspond to constraints as encoded in the NLP, not yet to continuous
    costates or continuous multiplier fields.
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
    """
    Continuous dual quantities represented on mesh and collocation nodes.
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
    """
    Full decoded collocation postprocessing payload.
    """
    layout: CollocationLayout

    primal: CollocationPrimalGrid
    primal_scaled: Optional[CollocationPrimalGrid] = None

    kkt: Optional[CollocationKKTMultipliers] = None
    kkt_scaled: Optional[CollocationKKTMultipliers] = None

    bound_kkt: Optional[CollocationBoundKKT] = None
    bound_kkt_scaled: Optional[CollocationBoundKKT] = None

    adjoint: Optional[CollocationAdjointGrid] = None

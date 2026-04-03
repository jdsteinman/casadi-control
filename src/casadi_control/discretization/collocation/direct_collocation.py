"""Direct collocation transcription.

This module provides :class:`~casadi_control.discretization.collocation.DirectCollocation`,
a facade that configures and constructs a direct-collocation nonlinear program (NLP)
from a continuous-time :class:`~casadi_control.problem.ocp.OCP`.

Only :class:`DirectCollocation` is considered public and stable. Internal helpers
that implement the transcription, initialization, postprocessing, and artifact
encoding live in sibling modules and may change without notice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Literal

import numpy as np

from ..base import Discretization, Guess, NLPLike, PostProcessed, Trajectory, DiscreteSolution, SolutionArtifact
from ...problem.ocp import OCP
from .schemes import make_table
from .transcription import build_collocation_nlp
from .initialize import guess_collocation
from .postprocess import postprocess_collocation
from .archive import collocation_to_artifact, collocation_from_artifact


def _as_1d_float_array(x: Any) -> np.ndarray:
    """Convert input to a one-dimensional float array."""
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array-like, got shape {arr.shape}")
    return arr


def _normalize_time_grid_to_s_mesh(
    t_mesh: np.ndarray,
    *,
    t0: float,
) -> np.ndarray:
    """
    Convert a physical time grid to a normalized grid s in [0,1] using:
        s = (t - t0) / (tN - t0)
    Requires strictly increasing t_mesh.
    """
    if t_mesh.size < 2:
        raise ValueError("time grid must have length >= 2")
    if not np.all(np.diff(t_mesh) > 0.0):
        raise ValueError("time grid must be strictly increasing")

    denom = float(t_mesh[-1] - t0)
    if denom <= 0.0:
        raise ValueError("time grid must satisfy t_mesh[-1] > t0")

    s = (t_mesh - float(t0)) / denom
    # Enforce exact endpoints (helps reproducibility)
    s[0] = 0.0
    s[-1] = 1.0
    return s


def _validate_s_mesh(s_mesh: np.ndarray) -> None:
    """Validate normalized mesh monotonicity and endpoint convention."""
    if s_mesh.size < 2:
        raise ValueError("s_mesh must have length >= 2")
    if not np.all(np.diff(s_mesh) > 0.0):
        raise ValueError("s_mesh must be strictly increasing")
    if abs(float(s_mesh[0]) - 0.0) > 1e-12 or abs(float(s_mesh[-1]) - 1.0) > 1e-12:
        raise ValueError("s_mesh must start at 0 and end at 1 (within tolerance)")


@dataclass(frozen=True)
class CollocationConfig:
    """Direct-collocation scheme configuration."""
    degree: int = 3
    scheme: str = "flgr"

    def __post_init__(self) -> None:
        if self.degree <= 0:
            raise ValueError("degree must be positive")
        if not self.scheme:
            raise ValueError("scheme must be non-empty")


class DirectCollocation(Discretization):
    """Direct collocation discretization frontend.

    This class configures a collocation scheme (table family + degree) and a mesh,
    and provides methods to:

    - build a solver-facing NLP (:meth:`build`)
    - generate an initial guess (:meth:`guess`)
    - decode raw solver output into trajectories (:meth:`postprocess`)
    - convert results to/from serializable artifacts (:meth:`to_artifact`, :meth:`from_artifact`)

    Parameters
    ----------
    N : int, optional
        Number of mesh intervals. Ignored when ``grid`` is provided.
    grid : array-like, optional
        User-specified mesh nodes. Interpreted according to ``grid_kind``.
    grid_kind : {"normalized", "physical"}, optional
        Interpretation of ``grid``:

        - ``"normalized"``: nodes are in the discretization coordinate ``s ∈ [0, 1]``.
        - ``"physical"``: nodes are physical-time nodes and are normalized internally.
    degree : int, optional
        Number of collocation points per mesh interval.
    scheme : str, optional
        Collocation table family identifier (e.g. ``"flgr"``).

    
    Methods
    -------
    build
    guess
    postprocess
    to_artifact
    from_artifact


    Notes
    -----
    This object does not solve the NLP. Solving is handled by the solver layer
    (e.g. :func:`casadi_control.solvers.solve_ipopt`) or the high-level
    :func:`casadi_control.solve` orchestration function.
    """

    name = "direct_collocation"

    def __init__(
        self,
        N: Optional[int] = None,
        grid: Optional[Any] = None,
        *,
        grid_kind: Literal["normalized", "physical"] = "normalized",
        degree: int = 3,
        scheme: str = "flgr",
    ):
        """Create a direct-collocation discretization configuration."""
        self.cfg = CollocationConfig(degree=int(degree), scheme=str(scheme))
        self._table = None

        self._grid_kind = str(grid_kind)
        self._grid = grid  # stored until build(), where we know ocp.t0
        self._N = None if N is None else int(N)

        if self._grid is None and self._N is None:
            raise ValueError("Provide either N or grid")

    @property
    def table(self):
        """Collocation coefficient table for the configured scheme/degree."""
        if self._table is None:
            self._table = make_table(self.cfg.scheme, self.cfg.degree)
        return self._table

    def _build_s_mesh(self, ocp: OCP) -> np.ndarray:
        if self._grid is None:
            assert self._N is not None
            s_mesh = np.linspace(0.0, 1.0, self._N + 1, dtype=float)
            _validate_s_mesh(s_mesh)
            return s_mesh

        grid = _as_1d_float_array(self._grid)

        if self._grid_kind == "normalized":
            s_mesh = grid
            _validate_s_mesh(s_mesh)
            return s_mesh

        if self._grid_kind == "physical":
            t_mesh = grid
            s_mesh = _normalize_time_grid_to_s_mesh(t_mesh, t0=float(getattr(ocp, "t0", 0.0)))
            _validate_s_mesh(s_mesh)
            return s_mesh

        raise ValueError(f"Unknown grid_kind={self._grid_kind!r}")

    def build(self, ocp: OCP):
        """Transcribe ``ocp`` into an NLP with the configured collocation scheme."""
        s_mesh = self._build_s_mesh(ocp)
        return build_collocation_nlp(
            ocp,
            s_mesh=s_mesh,
            table=self.table,
        )

    def guess(
        self,
        nlp: NLPLike,
        *,
        strategy: str = "default",
        prev: Optional[Trajectory] = None,
        **kwargs: Any,
    ) -> Guess:
        """Generate an initial guess for the collocation NLP."""
        return guess_collocation(nlp, strategy=strategy, prev=prev, **kwargs)

    def postprocess(self, ocp: OCP, nlp: NLPLike, sol: DiscreteSolution) -> PostProcessed:
        """Postprocess a raw collocation solution into trajectories."""
        return postprocess_collocation(ocp, nlp, sol)

    def to_artifact(self, sol: DiscreteSolution, pp: PostProcessed) -> SolutionArtifact:
        """Convert a solved result into a serializable artifact."""
        return collocation_to_artifact(sol, pp)

    def from_artifact(self, art: SolutionArtifact) -> PostProcessed:
        """Rebuild a plot-ready postprocessed result from an artifact."""
        return collocation_from_artifact(art)

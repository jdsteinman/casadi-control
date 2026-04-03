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

from ..base import (
    Discretization,
    Guess,
    NLPLike,
    PostProcessed,
    Trajectory,
    DiscreteSolution,
    SolutionArtifact,
)
from ...problem.ocp import OCP
from .common import as_1d_float_array, normalize_time_grid_to_s_mesh, validate_s_mesh
from .schemes import make_table
from .transcription import build_collocation_nlp
from .initialize import guess_collocation
from .postprocess import postprocess_collocation
from .archive import collocation_to_artifact, collocation_from_artifact


@dataclass(frozen=True)
class CollocationConfig:
    """Direct-collocation scheme configuration.

    Parameters
    ----------
    degree : int, optional
        Number of collocation points per mesh interval.
    scheme : str, optional
        Collocation table family identifier, such as ``"flgr"``.
    """
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

    Examples
    --------
    Build with a uniform normalized mesh:

    >>> tx = DirectCollocation(N=40, degree=3, scheme="flgr")
    >>> nlp = tx.build(ocp)

    Build with a user-defined physical-time mesh:

    >>> tx = DirectCollocation(grid=[0.0, 0.1, 0.3, 1.0], grid_kind="physical")
    >>> nlp = tx.build(ocp)

    Solve and postprocess:

    >>> guess = tx.guess(nlp, strategy="default")
    >>> sol = solve_ipopt(nlp, guess=guess)
    >>> pp = tx.postprocess(ocp, nlp, sol)
    >>> x_mid = pp.x(0.5 * pp.traj.tf)

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
        """Create a direct-collocation discretization configuration.

        Either ``N`` or ``grid`` must be supplied. Use ``N`` for a uniform mesh
        on the normalized interval and ``grid`` when you want explicit control
        over node placement.
        """
        self.cfg = CollocationConfig(degree=int(degree), scheme=str(scheme))
        self._table = None

        self._grid_kind = str(grid_kind)
        self._grid = grid  # stored until build(), where we know ocp.t0
        self._N = None if N is None else int(N)

        if self._grid is None and self._N is None:
            raise ValueError("Provide either N or grid")

    @property
    def table(self):
        """Collocation coefficient table for the configured scheme/degree.

        Most users do not need this property during ordinary solve workflows.
        It is mainly useful when inspecting the underlying transcription or
        reproducing coefficient tables in custom analyses.
        """
        if self._table is None:
            self._table = make_table(self.cfg.scheme, self.cfg.degree)
        return self._table

    def _build_s_mesh(self, ocp: OCP) -> np.ndarray:
        if self._grid is None:
            assert self._N is not None
            s_mesh = np.linspace(0.0, 1.0, self._N + 1, dtype=float)
            return validate_s_mesh(s_mesh)

        grid = as_1d_float_array(self._grid)

        if self._grid_kind == "normalized":
            return validate_s_mesh(grid)

        if self._grid_kind == "physical":
            t_mesh = grid
            return normalize_time_grid_to_s_mesh(
                t_mesh,
                t0=float(getattr(ocp, "t0", 0.0)),
            )

        raise ValueError(f"Unknown grid_kind={self._grid_kind!r}")

    def build(self, ocp: OCP):
        """Transcribe ``ocp`` into an NLP with the configured collocation scheme.

        The resulting :class:`~casadi_control.discretization.base.NLP` contains
        the decision-vector layout, default guess, bounds, and metadata needed
        by :meth:`guess` and :meth:`postprocess`.
        """
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
        """Generate an initial guess for the collocation NLP.

        Parameters
        ----------
        nlp : NLPLike
            NLP returned by :meth:`build`.
        strategy : {"default", "nlp", "blocks", "const", "functions", "prev"}, optional
            Guess construction strategy.
        prev : Trajectory, optional
            Previous postprocessed trajectory used by ``strategy="prev"``.
        **kwargs
            Strategy-specific options.

        Other Parameters
        ----------------
        tf : float
            Physical final time used by the ``"const"``, ``"functions"``, and
            ``"prev"`` strategies.
        x, u
            State/control values or callables, depending on the strategy.
        p
            Optional parameter guess.
        blocks : dict
            Decision-vector blocks in solver coordinates for
            ``strategy="blocks"``.
        mult_x0, mult_g0
            Optional warm-start multipliers used with ``strategy="prev"``.

        Returns
        -------
        Guess
            Flat collocation decision-vector guess plus strategy metadata in
            ``Guess.info``.

        Notes
        -----
        Strategy summary:

        - ``"default"`` or ``"nlp"`` returns the discretization-provided
          ``nlp.w0`` unchanged.
        - ``"blocks"`` lets you overwrite selected solver-space blocks such as
          ``X_mesh``, ``X_colloc``, ``U_colloc``, ``p``, or ``tf``.
        - ``"const"`` fills the mesh from constant physical values.
        - ``"functions"`` samples physical-time callables on the mesh and
          collocation nodes.
        - ``"prev"`` interpolates a previous :class:`Trajectory` onto the new
          mesh and is the standard continuation / mesh-refinement path.

        For ``"const"``, ``"functions"``, and ``"prev"``, inputs are specified
        in physical coordinates and converted to solver coordinates if the OCP
        uses scaling. For ``"blocks"``, values are written directly in solver
        coordinates.
        """
        return guess_collocation(nlp, strategy=strategy, prev=prev, **kwargs)

    def postprocess(self, ocp: OCP, nlp: NLPLike, sol: DiscreteSolution) -> PostProcessed:
        """Postprocess a raw collocation solution into trajectories.

        The returned :class:`PostProcessed` object is the main entry point for
        inspecting results:

        - ``pp.x(t)`` and ``pp.u(t)`` evaluate the primal solution
        - ``pp.diag`` reports mesh, degree, scaling, and residual information
        - ``pp.decoded`` exposes collocation-grid arrays for custom plotting or
          debugging
        - dual evaluators are available when the solver returned multipliers

        For direct collocation, ``pp.decoded`` is a
        :class:`~casadi_control.discretization.collocation.decode.CollocationDecoded`
        instance. Its most useful fields are:

        - ``decoded.layout``: mesh metadata such as ``N``, ``K``, ``tau``, and
          ``s_mesh``
        - ``decoded.primal``: physical-time state/control arrays on mesh and
          collocation nodes
        - ``decoded.primal_scaled``: the same primal arrays in solver/scaled
          coordinates when scaling is active
        - ``decoded.kkt`` / ``decoded.kkt_scaled``: decoded constraint
          multipliers on the NLP grid
        - ``decoded.bound_kkt`` / ``decoded.bound_kkt_scaled``: decoded
          decision-variable bound multipliers
        - ``decoded.adjoint``: interpreted costates and path/state multipliers
          on the collocation grid

        A good rule of thumb is:

        - use ``pp.x(...)`` and ``pp.u(...)`` for ordinary analysis and plotting
        - use ``pp.decoded`` when you need node values, raw multipliers, or
          scaled-versus-physical arrays
        """
        return postprocess_collocation(ocp, nlp, sol)

    def to_artifact(self, sol: DiscreteSolution, pp: PostProcessed) -> SolutionArtifact:
        """Convert a solved result into a serializable artifact.

        The artifact stores enough decoded collocation data to reconstruct a
        plot-ready :class:`PostProcessed` result later. A typical persistence
        workflow is:

        >>> art = tx.to_artifact(sol, pp)
        >>> save_npz("run.npz", art)

        See :func:`casadi_control.discretization.collocation.save_npz`.
        """
        return collocation_to_artifact(sol, pp)

    def from_artifact(self, art: SolutionArtifact) -> PostProcessed:
        """Rebuild a plot-ready postprocessed result from an artifact.

        This is intended for workflows such as offline plotting, regression
        comparisons, or loading previously solved runs without rebuilding and
        re-solving the NLP.

        >>> art = load_npz("run.npz")
        >>> pp = tx.from_artifact(art)
        """
        return collocation_from_artifact(art)

"""
Discretization base interfaces and scheme-agnostic containers.

The discretization layer converts a continuous-time optimal control problem into a
solver-facing nonlinear program (NLP). This module defines the minimal interfaces
and data containers shared by all transcription methods.

The key public types are

- ``NLP``: concrete CasADi-backed NLP container.
- ``Guess``: initial iterate and optional warm-start multipliers.
- ``DiscreteSolution``: solver-agnostic raw result.
- ``Trajectory`` / ``DualTrajectory`` / ``PostProcessed``: evaluation-first views.
- ``SolutionArtifact``: serializable payload for saving/loading results.
- ``Discretization``: abstract interface implemented by transcription methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np
import casadi as ca

if TYPE_CHECKING:
    from casadi_control.problem.ocp import OCP  # avoid runtime import cycles

Array = np.ndarray
TimeLike = float | Array


# =============================================================================
# NLP container (CasADi-backed, solver-facing)
# =============================================================================

@dataclass
class NLP:
    """Concrete nonlinear program container (CasADi-backed).

    Discretizations build an :class:`NLP`; solver adapters consume it.

    Parameters
    ----------
    prob : dict[str, casadi.MX]
        CasADi NLP dictionary following ``nlpsol`` conventions, typically
        ``{"x": w, "f": f, "g": g}`` with optional entries such as ``p``.
    lbx, ubx : ndarray
        Lower/upper bounds for the decision vector ``w``.
    lbg, ubg : ndarray
        Lower/upper bounds for the constraint vector ``g(w)``.
        For equality constraints, ``lbg[i] = ubg[i] = 0``.
    w0 : ndarray
        Default initial guess for the decision vector.
    meta : dict, optional
        Discretization-defined metadata used for decoding and postprocessing.
        Store index maps, grid information, scaling information, etc.
    unpack : casadi.Function, optional
        Optional decoding function. If provided, calling :meth:`unpack` returns a
        mapping ``{name: array}`` as specified by ``meta["unpack_outputs"]``.
        For example, this mapping can take the optimization vector and 
        return the state/control/parameter vectors.

    Attributes
    ----------
    n_w, n_g : int
        Sizes of decision and constraint vectors.

    Notes
    -----
    This container mirrors CasADi's NLP representation and is intentionally
    solver-facing.
    """

    prob: Dict[str, ca.MX]

    lbx: Array
    ubx: Array
    lbg: Array
    ubg: Array
    w0: Array

    meta: Dict[str, Any] = field(default_factory=dict)

    n_w: int = field(init=False)
    n_g: int = field(init=False)

    _unpack: Optional[ca.Function] = None
    _eval_f: ca.Function = field(init=False, repr=False)
    _eval_g: ca.Function = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.lbx = np.asarray(self.lbx, float).reshape(-1)
        self.ubx = np.asarray(self.ubx, float).reshape(-1)
        self.lbg = np.asarray(self.lbg, float).reshape(-1)
        self.ubg = np.asarray(self.ubg, float).reshape(-1)
        self.w0 = np.asarray(self.w0, float).reshape(-1)

        self.n_w = int(self.lbx.size)
        self.n_g = int(self.lbg.size)

        if self.ubx.size != self.n_w or self.w0.size != self.n_w:
            raise ValueError("Inconsistent decision-vector sizes: lbx/ubx/w0 mismatch.")
        if self.ubg.size != self.n_g:
            raise ValueError("Inconsistent constraint-vector sizes: lbg/ubg mismatch.")

        w = self.prob["x"]
        g = self.prob["g"]
        f = self.prob["f"]

        self._eval_g = ca.Function("eval_g", [w], [g])
        self._eval_f = ca.Function("eval_f", [w], [f])

    def eval_g(self, w: Array) -> Array:
        """Evaluate constraint vector g(w)."""
        wv = np.asarray(w, float).reshape(-1)
        return np.asarray(self._eval_g(wv), dtype=float).reshape(-1)

    def eval_f(self, w: Array) -> float:
        """Evaluate objective scalar f(w)."""
        wv = np.asarray(w, float).reshape(-1)
        return float(np.asarray(self._eval_f(wv), dtype=float).reshape(-1)[0])

    def unpack(self, w: Array) -> Dict[str, Array]:
        """Evaluate attached ``unpack(w)`` and return ``{name: array}``.

        Requires:
        - ``self.unpack`` is set, and
        - ``meta["unpack_outputs"]`` lists output names in the same order as the
          function outputs.
        """
        if self.unpack is None:
            raise RuntimeError("NLP has no unpack function attached.")

        names = self.meta.get("unpack_outputs", None)
        if not names:
            raise RuntimeError("nlp.meta['unpack_outputs'] missing or empty.")

        out = self._unpack(np.asarray(w, float).reshape(-1))
        if len(names) != len(out):
            raise RuntimeError("unpack_outputs length mismatch with unpack outputs.")

        return {str(k): np.asarray(v, float) for k, v in zip(names, out)}


@runtime_checkable
class NLPLike(Protocol):
    """Minimal solver-facing NLP protocol.

    Solver adapters should depend only on this protocol rather than a concrete
    :class:`~casadi_control.discretization.base.NLP` implementation.
    """

    n_w: int
    n_g: int
    lbx: Array
    ubx: Array
    lbg: Array
    ubg: Array
    w0: Array
    meta: Dict[str, Any]

    def eval_f(self, w: Array) -> float: ...
    def eval_g(self, w: Array) -> Array: ...
    def unpack(self, w: Array) -> Dict[str, Array]: ...


# =============================================================================
# Solver result and guesses
# =============================================================================

@dataclass(frozen=True)
class Guess:
    """Initial iterate for an NLP solve.

    Parameters
    ----------
    w0 : ndarray
        Initial guess for the decision vector ``w`` (flattened).
    mult_x0 : ndarray, optional
        Initial guess for bound multipliers (same length as ``w0``).
    mult_g0 : ndarray, optional
        Initial guess for constraint multipliers (same length as constraint vector).
    info : dict, optional
        Free-form metadata describing how the guess was generated.

    Notes
    -----
    Arrays are stored as one-dimensional ``float`` vectors.

    This object is the handoff between a discretization and a solver. Most
    workflows obtain a guess from :meth:`Discretization.guess`, but advanced
    users can also build a :class:`Guess` directly and pass it to
    :func:`casadi_control.solve` or a lower-level solver adapter.

    Typical usage patterns are:

    - start from the discretization default ``nlp.w0``
    - overwrite selected blocks of the decision vector
    - warm-start a repeated solve using ``mult_x0`` and ``mult_g0``
    """

    w0: Array
    mult_x0: Optional[Array] = None
    mult_g0: Optional[Array] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "w0", np.asarray(self.w0, float).reshape(-1))
        if self.mult_x0 is not None:
            object.__setattr__(self, "mult_x0", np.asarray(self.mult_x0, float).reshape(-1))
        if self.mult_g0 is not None:
            object.__setattr__(self, "mult_g0", np.asarray(self.mult_g0, float).reshape(-1))


@dataclass(frozen=True)
class DiscreteSolution:
    """Raw solver output (solver-agnostic).

    Parameters
    ----------
    w_opt : ndarray
        Optimal decision vector (flattened).
    f_opt : float
        Optimal objective value.
    mult_x : ndarray, optional
        Bound multipliers associated with ``w_opt``.
    mult_g : ndarray, optional
        Constraint multipliers associated with ``g(w_opt)``.
    status : str, optional
        Short status string suitable for logging/branching.
    stats : dict, optional
        Solver-provided statistics (iteration counts, timing, etc.).

    Notes
    -----
    Use :meth:`Discretization.postprocess` to obtain unpacked trajectories.
    """

    w_opt: Array
    f_opt: float
    mult_x: Optional[Array] = None
    mult_g: Optional[Array] = None
    status: str = "unknown"
    stats: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "w_opt", np.asarray(self.w_opt, float).reshape(-1))
        if self.mult_x is not None:
            object.__setattr__(self, "mult_x", np.asarray(self.mult_x, float).reshape(-1))
        if self.mult_g is not None:
            object.__setattr__(self, "mult_g", np.asarray(self.mult_g, float).reshape(-1))


# =============================================================================
# Post processing
# =============================================================================

@runtime_checkable
class Trajectory(Protocol):
    """Canonical postprocessed primal trajectory: evaluate x(t), u(t)."""

    nx: int
    nu: int
    tf: float

    def x(self, t: TimeLike) -> Array: ...
    def u(self, t: TimeLike) -> Array: ...


@runtime_checkable
class DualTrajectory(Protocol):
    """Canonical postprocessed dual trajectory."""

    nx: int
    nc: int
    ns: int
    tf: float

    def costate(self, t: TimeLike) -> Array: ...
    def path_multiplier(self, t: TimeLike) -> Array: ...
    def state_multiplier(self, t: TimeLike) -> Array: ...


@dataclass
class PostProcessed:
    """Scheme-agnostic postprocessing result.

    Parameters
    ----------
    traj
        Continuous primal evaluator. Use :meth:`x` and :meth:`u` for the most
        common evaluation path.
    dual_traj
        Continuous dual evaluator, if available.
    decoded
        Scheme-specific decoded discrete payload. This is useful when you need
        mesh-level arrays, collocation-node values, or other discretization
        details in addition to evaluator-style access.
    diag
        Free-form diagnostics.

    Notes
    -----
    :class:`PostProcessed` is the main analysis object returned after solving.
    It is designed so that most downstream code can stay independent of the
    underlying transcription:

    - call :meth:`x(t)` and :meth:`u(t)` to evaluate the primal solution
    - inspect ``diag`` for mesh, degree, residual, and solver summary data
    - use ``decoded`` when you need discretization-specific arrays for plotting,
      debugging, or custom postprocessing

    Think of ``decoded`` as the structured, grid-level companion to the
    evaluator-style ``traj`` object. Where ``traj`` answers "what is x(t)?",
    ``decoded`` answers questions like:

    - what are the state and control values at mesh and collocation nodes?
    - what normalized mesh and collocation table were used?
    - what raw NLP multipliers did the solver return?
    - what interpreted dual quantities were reconstructed during postprocessing?

    If dual information is available, :meth:`costate`,
    :meth:`path_multiplier`, and :meth:`state_multiplier` provide the same
    evaluator-style interface for interpreted dual trajectories.
    """

    traj: Trajectory
    dual_traj: Optional[DualTrajectory] = None
    decoded: Any = None
    diag: Dict[str, Any] = field(default_factory=dict)

    def x(self, t: TimeLike) -> np.ndarray:
        return np.asarray(self.traj.x(t), dtype=float)

    def u(self, t: TimeLike) -> np.ndarray:
        return np.asarray(self.traj.u(t), dtype=float)

    def costate(self, t: TimeLike) -> np.ndarray:
        if self.dual_traj is None:
            raise AttributeError("No costate trajectory available.")
        return np.asarray(self.dual_traj.costate(t), dtype=float)

    def path_multiplier(self, t: TimeLike) -> np.ndarray:
        if self.dual_traj is None:
            raise AttributeError("No path-multiplier trajectory available.")
        return np.asarray(self.dual_traj.path_multiplier(t), dtype=float)

    def state_multiplier(self, t: TimeLike) -> np.ndarray:
        if self.dual_traj is None:
            raise AttributeError("No state-multiplier trajectory available.")
        return np.asarray(self.dual_traj.state_multiplier(t), dtype=float)


# =============================================================================
# Serializable solution artifact (scheme-agnostic)
# =============================================================================

@dataclass(frozen=True)
class SolutionArtifact:
    """Serializable, scheme-agnostic solution payload.

    Artifacts allow saving/loading postprocessed results without requiring the
    original solver object. They are intended for reproducibility and plotting.

    Parameters
    ----------
    discretization : str
        Discretization identifier (e.g., ``"direct_collocation"``).
    arrays : dict[str, ndarray], optional
        Numpy arrays storing decoded data (numeric only).
    meta : dict, optional
        Small JSON-serializable metadata dictionary.

    Contract
    --------
    - ``arrays`` contains only numpy arrays (or empty arrays for missing data).
    - ``meta`` is JSON-serializable and should remain small.

    Notes
    -----
    A :class:`SolutionArtifact` is the portable representation you save to disk
    after a solve and later reload for plotting, comparison, or warm-start
    preparation. The usual pattern is:

    1. solve and postprocess an OCP
    2. call :meth:`Discretization.to_artifact`
    3. persist the artifact with a discretization-specific I/O helper
    4. reload it later and reconstruct :class:`PostProcessed` with
       :meth:`Discretization.from_artifact`
    """

    discretization: str

    arrays: Dict[str, Array] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Discretization API
# =============================================================================

class Discretization(ABC):
    """Abstract transcription/discretization interface.

    Each discretization is responsible for:

    - building an NLP from an OCP (:meth:`build`)
    - providing (optional) scheme-specific initial guesses (:meth:`guess`)
    - postprocessing raw solver output into evaluator-style trajectories (:meth:`postprocess`)
    - exporting/importing artifacts for reproducibility (:meth:`to_artifact`, :meth:`from_artifact`)

    In practice, the public workflow is:

    1. construct a discretization
    2. call :meth:`build` to obtain an :class:`NLP`
    3. optionally customize the initial guess with :meth:`guess`
    4. solve the NLP with a solver adapter
    5. call :meth:`postprocess` to obtain a :class:`PostProcessed` result
    6. optionally save/reload the result with :meth:`to_artifact` and
       :meth:`from_artifact`
    """

    name: str

    @abstractmethod
    def build(self, ocp: "OCP") -> NLP:
        """Transcribe the continuous-time OCP into a solver-facing NLP.

        The returned :class:`NLP` contains the CasADi problem dictionary, the
        bounds and default guess vectors expected by the solver, and metadata
        needed for decoding and postprocessing.
        """
        raise NotImplementedError

    def guess(
        self,
        nlp: NLPLike,
        *,
        strategy: str = "default",
        prev: Optional[Trajectory] = None,
        **kwargs: Any,
    ) -> Guess:
        """Construct an initial guess for the NLP.

        Discretizations may override this method to implement IVP rollouts,
        interpolation from previous solutions, continuation schedules, etc.

        Parameters
        ----------
        nlp : NLPLike
            NLP previously produced by :meth:`build`.
        strategy : str, optional
            Discretization-defined strategy name. The base implementation simply
            returns ``nlp.w0``.
        prev : Trajectory, optional
            Previous postprocessed trajectory used for continuation or mesh
            refinement.
        **kwargs
            Extra strategy-specific options.

        Returns
        -------
        Guess
            Initial decision vector and optional warm-start multipliers.

        Notes
        -----
        If you do not need a custom strategy, passing the returned
        :class:`Guess` directly to a solver is sufficient. For repeated solves,
        discretizations may accept ``prev`` plus scheme-specific keyword
        arguments to interpolate an earlier solution onto the new mesh.
        """
        info = {"strategy": strategy, "used_prev": prev is not None}
        return Guess(w0=np.asarray(nlp.w0, float).reshape(-1), info=info)

    @abstractmethod
    def postprocess(self, ocp: "OCP", nlp: NLPLike, sol: DiscreteSolution) -> PostProcessed:
        """Decode raw solver output into a postprocessed trajectory interface.

        Parameters
        ----------
        ocp : OCP
            Original problem definition.
        nlp : NLPLike
            NLP that was solved.
        sol : DiscreteSolution
            Raw solver output.

        Returns
        -------
        PostProcessed
            Evaluator-style primal/dual trajectories, decoded arrays, and
            diagnostics suitable for analysis, plotting, and artifact export.
        """
        raise NotImplementedError

    @abstractmethod
    def to_artifact(self, sol: DiscreteSolution, pp: PostProcessed) -> SolutionArtifact:
        """Convert a solved result into a serializable artifact.

        This method packages the information needed to reconstruct a
        :class:`PostProcessed` result without keeping the original solver
        objects alive. Persist the returned artifact with an appropriate helper
        for the concrete discretization.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement to_artifact().")

    @abstractmethod
    def from_artifact(self, art: SolutionArtifact) -> PostProcessed:
        """Rebuild a plot-ready postprocessed result from an artifact.

        This is the inverse of :meth:`to_artifact`. It is intended for
        workflows where a solved result is saved, transferred, or versioned and
        later reloaded for plotting or comparison.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement from_artifact().")


__all__ = [
    "NLP",
    "NLPLike",
    "Guess",
    "DiscreteSolution",
    "Trajectory",
    "DualTrajectory",
    "PostProcessed",
    "SolutionArtifact",
    "Discretization",
]

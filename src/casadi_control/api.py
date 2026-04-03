"""High-level orchestration API.

This module provides :func:`solve`, a convenience function that runs the
end-to-end pipeline:

``OCP`` → build NLP via a ``Discretization`` → solve via a solver adapter →
postprocess into trajectories.

Transcription and solving remain separate concerns: discretizations build NLPs,
solvers operate on NLPs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .problem import OCP
from .discretization import Discretization, Guess, DiscreteSolution
from .discretization.base import NLPLike, PostProcessed, Trajectory
from .solvers import solve_ipopt, IpoptOptions


@dataclass(frozen=True)
class SolveResult:
    """Unified solve output bundle.

    Attributes
    ----------
    ocp : OCP
        Problem definition used for this run.
    discretization : Discretization
        Discretization instance that produced the NLP.
    nlp : NLPLike
        Solver-facing nonlinear program used during the solve.
    guess : Guess
        Initial iterate passed to the solver.
    sol : DiscreteSolution
        Raw solver output.
    pp : PostProcessed
        Postprocessed trajectories and diagnostics.
    info : dict
        Convenience metadata for orchestration-level details.

    Notes
    -----
    The object supports legacy tuple unpacking:
    ``sol, pp = solve(...)``.
    """

    ocp: OCP
    discretization: Discretization
    nlp: NLPLike
    guess: Guess
    sol: DiscreteSolution
    pp: PostProcessed
    info: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Yield ``(sol, pp)`` for backward-compatible tuple unpacking."""
        yield self.sol
        yield self.pp

    def as_tuple(self) -> Tuple[DiscreteSolution, PostProcessed]:
        """Return ``(sol, pp)`` tuple."""
        return self.sol, self.pp


def solve(
    ocp: OCP,
    discretization: Discretization,
    *,
    solver: str = "ipopt",
    guess: Optional[Guess] = None,
    guess_strategy: str = "default",
    guess_prev: Optional[Trajectory] = None,
    guess_kwargs: Optional[Dict[str, Any]] = None,
    solver_opts: Optional[IpoptOptions] = None,
    casadi_opts: Optional[Dict[str, Any]] = None,
    ipopt_opts: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None,
    solver_name: str = "ipopt",
) -> SolveResult:
    """Build, solve, and postprocess an OCP in one call."""
    nlp: NLPLike = discretization.build(ocp)
    used_explicit_guess = guess is not None

    if guess is None:
        kwargs = {} if guess_kwargs is None else dict(guess_kwargs)
        guess = discretization.guess(
            nlp,
            strategy=guess_strategy,
            prev=guess_prev,
            **kwargs,
        )

    if solver.lower() != "ipopt":
        raise ValueError(f"Unsupported solver {solver!r}. Supported: 'ipopt'.")

    sol = solve_ipopt(
        nlp,
        guess=guess,
        opts=solver_opts,
        casadi_opts=casadi_opts,
        ipopt_opts=ipopt_opts,
        options=options,
        solver_name=solver_name,
    )
    pp = discretization.postprocess(ocp, nlp, sol)
    return SolveResult(
        ocp=ocp,
        discretization=discretization,
        nlp=nlp,
        guess=guess,
        sol=sol,
        pp=pp,
        info={
            "solver": solver.lower(),
            "guess_strategy": "explicit_guess" if used_explicit_guess else guess_strategy,
        },
    )


__all__ = ["SolveResult", "solve"]

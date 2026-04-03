"""
CasADi-Control: Optimal Control in Python
=====================================

`casadi_control` provides tools for defining, discretizing, and solving
continuous-time optimal control problems (OCPs) with CasADi.

Subpackages
-----------

problem
    OCP model definition and scaling.
discretization
    Transcription interfaces, collocation implementation, and factory helpers.
solvers
    NLP solver adapters and option containers.

Main Interfaces
---------------

Problem definition
    OCP
    Scaling

Discretization
    Discretization
    DirectCollocation
    DiscretizationFactory
    available_discretizations

Solve API
    solve
    SolveResult
    solve_ipopt
    IpoptOptions

Data containers

    NLP
    Guess
    DiscreteSolution
    SolutionArtifact

Quick Start
-----------

>>> from casadi_control import OCP, DiscretizationFactory, solve
>>> # ocp = OCP(...)
>>> disc = DiscretizationFactory("collocation", N=40, degree=3, scheme="flgr")
>>> # result = solve(ocp, disc)

"""

from .problem import OCP, Scaling
from .discretization import (
    NLP,
    Guess,
    DiscreteSolution,
    SolutionArtifact,
    Discretization,
    DirectCollocation,
    DiscretizationFactory,
    available_discretizations,
)
from .solvers import solve_ipopt, IpoptOptions
from .api import SolveResult, solve

__all__ = [
    "OCP",
    "Scaling",
    "NLP",
    "Guess",
    "DiscreteSolution",
    "SolutionArtifact",
    "Discretization",
    "DirectCollocation",
    "DiscretizationFactory",
    "available_discretizations",
    "IpoptOptions",
    "SolveResult",
    "solve_ipopt",
    "solve",
]

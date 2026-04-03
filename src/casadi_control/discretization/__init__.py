"""Discretization interfaces and default scheme factories.

This namespace provides scheme-agnostic NLP/solution containers, the base
discretization interface, and the default direct-collocation implementation.
It also exposes a factory-oriented API for constructing discretizations by
name.
"""

from .base import NLP, Guess, DiscreteSolution, SolutionArtifact, Discretization
from .collocation import DirectCollocation
from .registry import (
    DiscretizationRegistry,
    DiscretizationFactory,
    available_discretizations,
)

__all__ = [
    "NLP",
    "Guess",
    "DiscreteSolution",
    "SolutionArtifact",
    "Discretization",
    "DirectCollocation",
    "DiscretizationRegistry",
    "DiscretizationFactory",
    "available_discretizations",
]

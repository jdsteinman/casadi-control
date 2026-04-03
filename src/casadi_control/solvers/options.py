"""Typed solver option containers.

This module provides small immutable configuration objects for NLP solvers.
The primary container is :class:`IpoptOptions`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class IpoptOptions:
    """Structured options container for IPOPT via CasADi.

    This object separates CasADi ``nlpsol`` options from IPOPT plugin options.

    Parameters
    ----------
    casadi : dict, optional
        Top-level CasADi ``nlpsol`` options (e.g. ``{"print_time": False}``).
    ipopt : dict, optional
        IPOPT plugin options (e.g. tolerances, linear solver).
    legacy : dict, optional
        Backward-compatible mixed options dictionary (if you still support it).
    solver_name : str, optional
        Name passed to ``ca.nlpsol(name, "ipopt", ...)``.

    Examples
    --------
    .. code-block:: python

        from casadi_control.solvers import IpoptOptions

        opts = IpoptOptions(
            ipopt={"tol": 1e-8, "linear_solver": "mumps"},
            casadi={"print_time": False},
        )
    """

    casadi: Dict[str, Any] = field(default_factory=dict)
    ipopt: Dict[str, Any] = field(default_factory=dict)
    legacy: Optional[Dict[str, Any]] = None
    solver_name: str = "ipopt"


__all__ = ["IpoptOptions"]

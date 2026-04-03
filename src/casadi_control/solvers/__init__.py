"""Solver adapters and option containers for discretized OCP NLPs."""

from .ipopt import solve_ipopt
from .options import IpoptOptions

__all__ = ["solve_ipopt", "IpoptOptions"]

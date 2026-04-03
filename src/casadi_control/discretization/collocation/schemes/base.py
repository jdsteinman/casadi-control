"""Base data structures for collocation coefficient tables."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class CollocationTable:
    """
    Collocation scheme data on the reference interval [0, 1].

    Conventions:
      - tau[0] = 0, tau[1:K] are collocation points.
      - c maps values of a polynomial at `tau` (K+1) to
        the polynomials value at t=1.
      - D maps values of a polynomial at `tau` (K+1) to 
        values of its derivative at `tau` (K+1).
      - w maps values of a function at `tau[1:K]` to 
        to associated quadrature on [0,1] with weight 1.
    """
    name: str
    degree: int

    tau: Array          # (K+1,)
    c: Array            # (K+1,)
    D: Array            # (K, K+1) in current FLGR implementation
    w: Array            # (K,)


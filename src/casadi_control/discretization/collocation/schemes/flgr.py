"""Flipped Legendre-Gauss-Radau (FLGR) collocation coefficients.

The implementation uses Lagrange basis polynomials on nodes
``tau = [0, tau_1, ..., tau_K]`` where ``tau_1..tau_K`` are CasADi's Radau
nodes mapped to ``[0, 1]``.
"""
from __future__ import annotations

import numpy as np
import casadi as ca

from .base import CollocationTable

def flgr_table(degree: int):
    """Build FLGR collocation coefficients on the unit interval.

    Parameters
    ----------
    degree : int
        Number of collocation nodes per mesh interval.

    Returns
    -------
    CollocationTable
        Table with nodes ``tau``, endpoint interpolation coefficients ``c``,
        derivative matrix ``D``, and quadrature weights ``w``.

    Notes
    -----
    ``D`` has shape ``(degree, degree + 1)`` and is evaluated at collocation
    nodes only. ``w`` stores quadrature weights for collocation nodes
    ``tau[1:]``.
    """
    # Collocation nodes on [0, 1], prepended with 0
    tau = np.append(0.0, ca.collocation_points(degree, "radau"))

    b = np.zeros(degree + 1)
    c = np.zeros(degree + 1)
    D = np.zeros((degree, degree + 1))

    for r in range(degree + 1):
        poly_lr = np.poly1d([1.0])
        for s in range(degree + 1):
            if s != r:
                poly_lr *= np.poly1d([1.0, -tau[s]]) / (tau[r] - tau[s])

        c[r] = poly_lr(1.0)

        poly_lr_der = np.polyder(poly_lr)
        for j in range(degree):
            D[j, r] = poly_lr_der(tau[j + 1])

        poly_lr_int = np.polyint(poly_lr)
        b[r] = poly_lr_int(1.0)

    b = b[1:]

    return CollocationTable(
        name="flgr",
        degree=degree,
        tau=tau,
        c=c,
        D=D,
        w=b,
    )


# examples/hager_lq_ocp.py
"""
Example adapted from Hager (2000), Numerische Mathematik, 87:247–282.

    minimize_{x(·), u(·)}
        J = ∫_0^1 ( x(t)^2 + 1/2 * u(t)^2 ) dt

    subject to
        x'(t) = 1/2 * x(t) + u(t),    t ∈ [0, 1],
        x(0)  = 1.

The optimal state, control, and costate are

    x*(t) =  ( 2 e^{3t} + e^{3} ) / ( 2 e^{3/2} + e^3 )
    u*(t) =  2 ( e^{3t} − e^{3} ) / ( 2 e^{3/2} + e^3 )
    λ*(t) = -2 ( e^{3t} - e^{3} ) / ( 2 e^{3/2} + e^3 )
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from casadi_control.problem import OCP, Scaling
from casadi_control.discretization import DirectCollocation
from casadi_control.solvers import solve_ipopt


# =============================================================================
# Problem definition
# =============================================================================

def build_lq_ocp() -> OCP:
    n_x, n_u = 1, 1

    def f_dyn(x, u, p, t):
        return 0.5 * x + u

    def l_run(x, u, p, t):
        return x**2 + 0.5 * u**2

    def l_end(x0, xf, p, t0, tf):
        return 0.0

    return OCP(
        n_x=n_x,
        n_u=n_u,
        n_p=0,
        t0=0.0,
        tf=1.0,
        f_dyn=f_dyn,
        l_run=l_run,
        l_end=l_end,
        x_bounds=(np.array([-np.inf]), np.array([np.inf])),
        u_bounds=(np.array([-np.inf]), np.array([np.inf])),
        x0_fixed=np.array([1.0]),
    )


# =============================================================================
# Analytical solution
# =============================================================================

def analytical_solution(t):
    t = np.asarray(t, float)
    denom = np.exp(3 * t / 2) * (2 + np.exp(3))
    x = (2 * np.exp(3 * t) + np.exp(3)) / denom
    u = 2 * (np.exp(3 * t) - np.exp(3)) / denom
    lam = 2 * (-np.exp(3 * t) + np.exp(3)) / denom
    return x, u, lam


# =============================================================================
# Main
# =============================================================================

def main(
    grid = None,
    N: int = 40,
    degree: int = 3,
) -> None:
    # -------------------------------------------------------------------------
    # Build and (optionally) scale the OCP
    # -------------------------------------------------------------------------
    ocp = build_lq_ocp()

    scaling = Scaling(x_ref=2.0, u_ref=0.5, t_ref=2.0, J_ref=2.0)
    ocp_scaled = ocp.scaled(scaling)

    # -------------------------------------------------------------------------
    # Discretize (new options: control strategy + optional penalties)
    # -------------------------------------------------------------------------
    if grid is not None:
        tx = DirectCollocation(
            grid=grid,
            degree=degree,
        )
    elif N is not None:
        tx = DirectCollocation(
            N=N,
            degree=degree,
        )
    else:
        raise ValueError("Must supply grid points or num. intervals")
    nlp = tx.build(ocp_scaled)

    # -------------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------------
    sol = solve_ipopt(nlp)

    # Postprocess in physical units (important: pass the physical OCP here)
    pp = tx.postprocess(ocp, nlp, sol)

    # =============================================================================
    # Numerical trajectories
    # =============================================================================
    t_nodes = pp.decoded.primal.t_nodes
    x_num = pp.x(t_nodes).reshape(-1)
    lam_num = pp.costate(t_nodes).reshape(-1)

    # For control, sample on collocation times (and mesh times if ZOH is used)
    t_col = pp.decoded.primal.t_colloc.reshape(-1)
    u_num_col = pp.u(t_col).reshape(-1)

    # =============================================================================
    # Analytical trajectories
    # =============================================================================
    x_exact, _, lam_exact = analytical_solution(t_nodes)
    _, u_exact_col, _ = analytical_solution(t_col)

    # =============================================================================
    # Errors (∞-norm on sampling grids)
    # =============================================================================
    err_x = np.linalg.norm(x_num - x_exact, np.inf)
    err_u = np.linalg.norm(u_num_col - u_exact_col, np.inf)
    err_lam = np.linalg.norm(lam_num - lam_exact, np.inf)

    print(f"||x_num - x_exact||_inf   = {err_x:.3e}")
    print(f"||u_num - u_exact||_inf   = {err_u:.3e}   (evaluated at collocation times)")
    print(f"||lam_num - lam_exact||_inf = {err_lam:.3e}")

    # =============================================================================
    # Plots
    # =============================================================================
    plt.figure(figsize=(6, 4))
    plt.plot(t_nodes, x_num, "-", label="numerical")
    plt.plot(t_nodes, x_exact, "--", label="analytical")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.title("State")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(6, 4))
    plt.plot(t_col, u_num_col, "-", label="numerical (collocation times)")
    plt.plot(t_col, u_exact_col, "--", label="analytical (collocation times)")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u(t)$")
    plt.title("Control")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(6, 4))
    plt.plot(t_nodes, lam_num, "-", label="numerical")
    plt.plot(t_nodes, lam_exact, "--", label="analytical")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\lambda(t)$")
    plt.title("Costate")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    N = 100
    x = np.array([np.cos(k*np.pi/N) for k in range(N+1)])
    x = (x + 1) / 2
    x = np.flip(x)
    main(grid=x, degree=3)
#    main(N=N, degree=3)

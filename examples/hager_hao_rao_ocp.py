# examples/hager_lq_ocp.py
"""
Example adapted from 
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from casadi_control.problem import OCP, Scaling
from casadi_control.discretization import DirectCollocation
from casadi_control.solvers import solve_ipopt


def analytical_solution(t):
    t = np.asarray(t, float)
    x1 = (np.cosh(1-t) * (2*np.exp(3*t) + np.exp(3))) \
        / ((2 + np.exp(3))*np.exp(3*t/2)*np.cosh(1))
    x2 = np.cosh(1) / np.cosh(1-t)
    u1 = (2 * (np.exp(3*t) - np.exp(3))) \
        / ((2 + np.exp(3)) * np.exp(3*t/2))
    u2 = -np.cosh(1-t) * (np.tanh(1-t) + 0.5) / np.cosh(1)

    x = np.column_stack((x1, x2))
    u = np.column_stack((u1, u2))
    return x, u


def main(
    grid = None,
    N: int = 40,
    degree: int = 3,
) -> None:
    # Build ocp
    n_x, n_u = 2, 2

    def f_dyn(x, u, p, t):
        x1 = x[0]
        x2 = x[1]
        u1 = u[0]
        u2 = u[1]
        return ca.vertcat(
            x1 + u1 / x2 + u2 * x1 * x2, 
            -x2 * (0.5 + u2 * x2)
        )

    def l_run(x, u, p, t):
        x1 = x[0]
        x2 = x[1]
        u1 = u[0]
        u2 = u[1]
        return 2 * x1**2 * x2**2 + 1.25 / x2**2 + u2 / x2 + u1**2 + u2**2

    ocp = OCP(
        n_x=n_x,
        n_u=n_u,
        n_p=0,
        t0=0.0,
        tf=1.0,
        f_dyn=f_dyn,
        l_run=l_run,
        x0_fixed=np.array([1.0, 1.0]),
    )

    # Discretize
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

    nlp = tx.build(ocp)

    # Construct initial guess (zeros results in nan)
    guess = tx.guess(
        nlp,
        strategy="const",
        x=np.array([1.0, 1.0]),
        u=np.array([1.0, 1.0]),
        tf=1.0
    )
#    guess = tx.guess(
#        nlp,
#        strategy="functions",
#        x=x_init,
#        u=u_init,
#        tf=1.0
#    )

    # Solve
    ipopt_opts = {"tol": 1e-10}
    sol = solve_ipopt(nlp, guess=guess, ipopt_opts=ipopt_opts)

    # Postprocess in physical units (important: pass the physical OCP here)
    pp = tx.postprocess(ocp, nlp, sol)

    # Numerical trajectories
    t_nodes = pp.decoded.primal.t_nodes

    x_num = pp.x(t_nodes)
    u_num = pp.u(t_nodes)

    # Analytical trajectories
    x_exact, u_exact = analytical_solution(t_nodes)

    # Errors
    err_x = np.linalg.norm(x_num - x_exact, np.inf)
    err_u = np.linalg.norm(u_num - u_exact, np.inf)

    print(f"||x_num - x_exact||_inf   = {err_x:.3e}")
    print(f"||u_num - u_exact||_inf   = {err_u:.3e}")

    # Plots
    plt.figure(figsize=(6, 4))
    plt.plot(t_nodes, x_num[:,0], "-", label=r"$x_1$ (num)")
    plt.plot(t_nodes, x_exact[:,0], "--", label=r"$x_1$ (ex)")
    plt.plot(t_nodes, x_num[:,1], "-", label=r"$x_2$ (num)")
    plt.plot(t_nodes, x_exact[:,1], "--", label=r"$x_2$ (ex)")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x(t)$")
    plt.title("State")
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(6, 4))
    plt.plot(t_nodes, u_num[:,0], "-", label=r"$u_1$ (num)")
    plt.plot(t_nodes, u_exact[:,0], "--", label=r"$u_1$ (ex)")
    plt.plot(t_nodes, u_num[:,1], "-", label=r"$u_2$ (num)")
    plt.plot(t_nodes, u_exact[:,1], "--", label=r"$u_2$ (ex)")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$u(t)$")
    plt.title("Control")
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

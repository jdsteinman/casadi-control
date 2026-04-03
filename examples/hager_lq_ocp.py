# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hager Linear-Quadratic Example

# %% [raw] raw_mimetype="text/restructuredtext"
# This notebook solves Problem 1 from :cite:p:`hager1976`.

# %% [markdown]
# The continuous-time problem is
#
# $$
# \begin{aligned}
# \min_{x(\cdot),u(\cdot)} \quad &
# \int_0^1 \left(x(t)^2 + \tfrac{1}{2}u(t)^2\right)\,dt \\
# \text{s.t.}\quad &
# \dot{x}(t) = \tfrac{1}{2}x(t) + u(t), \qquad t \in [0,1], \\
# & x(0)=1.
# \end{aligned}
# $$
#
# The exact solution is
#
# $$
# x^\star(t)=\frac{2e^{3t}+e^3}{(2e^{3/2}+e^3)e^{3t/2}}, \qquad
# u^\star(t)=\frac{2(e^{3t}-e^3)}{(2e^{3/2}+e^3)e^{3t/2}},
# $$
#
# and the corresponding costate is
#
# $$
# \lambda^\star(t)=\frac{2(-e^{3t}+e^3)}{(2e^{3/2}+e^3)e^{3t/2}}.
# $$
#
# We solve the problem with direct collocation and compare the numerical
# state, control, and costate trajectories with these analytical
# expressions.
#
# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from casadi_control.problem import OCP, Scaling
from casadi_control.discretization import DirectCollocation
from casadi_control.solvers import solve_ipopt


def build_lq_ocp() -> OCP:
    def f_dyn(x, u, p, t):
        return 0.5 * x + u

    def l_run(x, u, p, t):
        return x**2 + 0.5 * u**2

    def l_end(x0, xf, p, t0, tf):
        return 0.0

    return OCP(
        n_x=1,
        n_u=1,
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


def analytical_solution(t):
    t = np.asarray(t, float)
    denom = np.exp(3 * t / 2) * (2 + np.exp(3))
    x = (2 * np.exp(3 * t) + np.exp(3)) / denom
    u = 2 * (np.exp(3 * t) - np.exp(3)) / denom
    lam = 2 * (-np.exp(3 * t) + np.exp(3)) / denom
    return x, u, lam


# %% [markdown]
# ## Build, scale, and discretize the OCP

# %%
ocp = build_lq_ocp()
scaling = Scaling(x_ref=[2.0], u_ref=[0.5], t_ref=2.0, J_ref=2.0)
ocp_scaled = ocp.scaled(scaling)

tx = DirectCollocation(N=40, degree=3, scheme="flgr")
nlp = tx.build(ocp_scaled)

# %% [markdown]
# ## Solve and postprocess

# %%
sol = solve_ipopt(nlp)
pp = tx.postprocess(ocp, nlp, sol)

t_nodes = pp.decoded.primal.t_nodes
t_col = pp.decoded.primal.t_colloc.reshape(-1)

x_num = pp.x(t_nodes).reshape(-1)
u_num = pp.u(t_col).reshape(-1)
lam_num = pp.costate(t_nodes).reshape(-1)

x_exact, _, lam_exact = analytical_solution(t_nodes)
_, u_exact, _ = analytical_solution(t_col)

err_x = np.linalg.norm(x_num - x_exact, np.inf)
err_u = np.linalg.norm(u_num - u_exact, np.inf)
err_lam = np.linalg.norm(lam_num - lam_exact, np.inf)

print(f"||x_num - x_exact||_inf   = {err_x:.3e}")
print(f"||u_num - u_exact||_inf   = {err_u:.3e}")
print(f"||lam_num - lam_exact||_inf = {err_lam:.3e}")

# %% [markdown]
# ## Plot the trajectories

# %%
fig, axes = plt.subplots(3, 1, figsize=(7, 10), constrained_layout=True)

axes[0].plot(t_nodes, x_num, "-", label="numerical")
axes[0].plot(t_nodes, x_exact, "--", label="analytical")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x(t)")
axes[0].set_title("State")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_col, u_num, "-", label="numerical")
axes[1].plot(t_col, u_exact, "--", label="analytical")
axes[1].set_xlabel("t")
axes[1].set_ylabel("u(t)")
axes[1].set_title("Control")
axes[1].grid(True)
axes[1].legend()

axes[2].plot(t_nodes, lam_num, "-", label="numerical")
axes[2].plot(t_nodes, lam_exact, "--", label="analytical")
axes[2].set_xlabel("t")
axes[2].set_ylabel("lambda(t)")
axes[2].set_title("Costate")
axes[2].grid(True)
axes[2].legend()

plt.show()

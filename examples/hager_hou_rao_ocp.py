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
# # Hager-Hou-Rao Example

# %% [raw] raw_mimetype="text/restructuredtext"
# This notebook reproduces equation (57) from :cite:p:`hager-hou-rao2016`.

# %% [markdown]
# The benchmark problem is
#
# $$
# \begin{aligned}
# \min_{x(\cdot),u(\cdot)} \quad &
# \int_0^1 \left(
# 2x_1(t)^2x_2(t)^2 + \frac{5}{4x_2(t)^2}
# + \frac{u_2(t)}{x_2(t)} + u_1(t)^2 + u_2(t)^2
# \right)\,dt \\
# \text{s.t.}\quad &
# \dot{x}_1(t)=x_1(t) + \frac{u_1(t)}{x_2(t)} + u_2(t)x_1(t)x_2(t), \\
# &
# \dot{x}_2(t)=-x_2(t)\left(\tfrac{1}{2}+u_2(t)x_2(t)\right), \\
# &
# x_1(0)=1,\qquad x_2(0)=1.
# \end{aligned}
# $$
#
# The exact solution is
#
# $$
# x_1^\star(t)=
# \frac{\cosh(1-t)\left(2e^{3t}+e^3\right)}
# {(2+e^3)e^{3t/2}\cosh(1)},
# \qquad
# x_2^\star(t)=\frac{\cosh(1)}{\cosh(1-t)},
# $$
#
# $$
# u_1^\star(t)=
# \frac{2(e^{3t}-e^3)}
# {(2+e^3)e^{3t/2}},
# \qquad
# u_2^\star(t)=
# -\frac{\cosh(1-t)\left(\tanh(1-t)+\tfrac{1}{2}\right)}{\cosh(1)}.
# $$
#
# We solve the problem with direct collocation and compare the numerical
# trajectories against these analytical expressions.
#
# %%
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from casadi_control.problem import OCP
from casadi_control.discretization import DirectCollocation
from casadi_control.solvers import solve_ipopt


def analytical_solution(t):
    t = np.asarray(t, float)
    x1 = (
        np.cosh(1 - t) * (2 * np.exp(3 * t) + np.exp(3))
        / ((2 + np.exp(3)) * np.exp(3 * t / 2) * np.cosh(1))
    )
    x2 = np.cosh(1) / np.cosh(1 - t)
    u1 = 2 * (np.exp(3 * t) - np.exp(3)) / ((2 + np.exp(3)) * np.exp(3 * t / 2))
    u2 = -np.cosh(1 - t) * (np.tanh(1 - t) + 0.5) / np.cosh(1)

    x = np.column_stack((x1, x2))
    u = np.column_stack((u1, u2))
    return x, u


def build_ocp() -> OCP:
    def f_dyn(x, u, p, t):
        x1 = x[0]
        x2 = x[1]
        u1 = u[0]
        u2 = u[1]
        return ca.vertcat(
            x1 + u1 / x2 + u2 * x1 * x2,
            -x2 * (0.5 + u2 * x2),
        )

    def l_run(x, u, p, t):
        x1 = x[0]
        x2 = x[1]
        u1 = u[0]
        u2 = u[1]
        return 2 * x1**2 * x2**2 + 1.25 / x2**2 + u2 / x2 + u1**2 + u2**2

    return OCP(
        n_x=2,
        n_u=2,
        n_p=0,
        t0=0.0,
        tf=1.0,
        f_dyn=f_dyn,
        l_run=l_run,
        x0_fixed=np.array([1.0, 1.0]),
    )


# %% [markdown]
# ## Build the OCP and discretization

# %%
ocp = build_ocp()
tx = DirectCollocation(N=40, degree=3, scheme="flgr")
nlp = tx.build(ocp)

guess = tx.guess(
    nlp,
    strategy="const",
    x=np.array([1.0, 1.0]),
    u=np.array([1.0, 1.0]),
    tf=1.0,
)

# %% [markdown]
# ## Solve and compare against the analytical solution

# %%
sol = solve_ipopt(nlp, guess=guess, ipopt_opts={"tol": 1e-10})
pp = tx.postprocess(ocp, nlp, sol)

t_nodes = pp.decoded.primal.t_nodes
x_num = pp.x(t_nodes)
u_num = pp.u(t_nodes)

x_exact, u_exact = analytical_solution(t_nodes)

err_x = np.linalg.norm(x_num - x_exact, np.inf)
err_u = np.linalg.norm(u_num - u_exact, np.inf)

print(f"||x_num - x_exact||_inf   = {err_x:.3e}")
print(f"||u_num - u_exact||_inf   = {err_u:.3e}")

# %% [markdown]
# ## Plot the trajectories

# %%
fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)

axes[0].plot(t_nodes, x_num[:, 0], "-", label=r"$x_1$ (num)")
axes[0].plot(t_nodes, x_exact[:, 0], "--", label=r"$x_1$ (exact)")
axes[0].plot(t_nodes, x_num[:, 1], "-", label=r"$x_2$ (num)")
axes[0].plot(t_nodes, x_exact[:, 1], "--", label=r"$x_2$ (exact)")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x(t)")
axes[0].set_title("State")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(t_nodes, u_num[:, 0], "-", label=r"$u_1$ (num)")
axes[1].plot(t_nodes, u_exact[:, 0], "--", label=r"$u_1$ (exact)")
axes[1].plot(t_nodes, u_num[:, 1], "-", label=r"$u_2$ (num)")
axes[1].plot(t_nodes, u_exact[:, 1], "--", label=r"$u_2$ (exact)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("u(t)")
axes[1].set_title("Control")
axes[1].grid(True)
axes[1].legend()

plt.show()

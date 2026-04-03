Problem Setup
=============

This section covers how to define a continuous-time OCP with
:class:`casadi_control.OCP`.

Model structure
---------------

An ``OCP`` instance defines:

* dimensions: ``n_x``, ``n_u``, optional ``n_p``
* time horizon: ``t0`` and ``tf`` (fixed or free-final-time bounds)
* dynamics: ``f_dyn(x, u, p, t)``
* objective: ``l_run(x, u, p, t)`` and/or ``l_end(x0, xf, p, t0, tf)``
* optional constraints: ``bnd_constr``, ``path_constr``, ``state_constr``
* optional bounds and initial/nominal values

Callback conventions
--------------------

Use CasADi-compatible callbacks:

* ``f_dyn`` returns shape ``(n_x, 1)``
* ``l_run`` and ``l_end`` return scalar-like outputs
* constraint callbacks return column vectors ``(m, 1)`` or empty vectors

Example
-------

.. code-block:: python

   import casadi as ca
   import numpy as np
   from casadi_control import OCP

   def f_dyn(x, u, p, t):
       return ca.vertcat(x[1], u[0])

   def l_run(x, u, p, t):
       return u[0] ** 2

   def l_end(x0, xf, p, t0, tf):
       return 100.0 * (xf[0] - 1.0) ** 2 + 10.0 * (xf[1] ** 2)

   ocp = OCP(
       n_x=2,
       n_u=1,
       t0=0.0,
       tf=1.0,
       f_dyn=f_dyn,
       l_run=l_run,
       l_end=l_end,
       x_bounds=(np.array([-5.0, -5.0]), np.array([5.0, 5.0])),
       u_bounds=(np.array([-2.0]), np.array([2.0])),
       x0_fixed=np.array([0.0, 0.0]),
   )

   ocp.validate()

Free-final-time problems
------------------------

Set ``tf=(lb, ub)`` to optimize the final time:

.. code-block:: python

   ocp = OCP(
       n_x=2,
       n_u=1,
       tf=(0.2, 3.0),
       f_dyn=f_dyn,
       l_run=l_run,
   )

Reference
---------

For full parameter-level API details, see
:mod:`casadi_control.problem.ocp`.

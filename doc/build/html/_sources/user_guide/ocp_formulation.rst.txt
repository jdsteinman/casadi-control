Optimal Control Formulation
===========================

CasADi-Control represents continuous-time optimal control problems in
the form

.. math::

    \begin{aligned}
    \min_{x,u,p}\quad
    & \phi(x(t_0),x(t_f),p)
      + \int_{t_0}^{t_f} \ell(x(t),u(t),p,t)\,dt \\
    \text{s.t.}\quad
    & x'(t) = f(x(t),u(t),p,t) \\
    & c(x(t),u(t),p,t) \le 0 \\
    & s(x(t),p,t) \le 0 \\
    & b(x(t_0),x(t_f)) = 0
    \end{aligned}

where

* :math:`x(t)` is the state
* :math:`u(t)` is the control
* :math:`p` are constant parameters

Mapping to the Library API
--------------------------

The mathematical objects correspond to the following callbacks:

+----------------------+----------------------------+
| Mathematical object  | Library callback           |
+======================+============================+
| :math:`f`            | ``f_dyn``                  |
+----------------------+----------------------------+
| :math:`\ell`         | ``l_run``                  |
+----------------------+----------------------------+
| :math:`\phi`         | ``l_end``                  |
+----------------------+----------------------------+
| :math:`c`            | ``path_constr``            |
+----------------------+----------------------------+
| :math:`s`            | ``state_constr``           |
+----------------------+----------------------------+
| :math:`b`            | ``bnd_constr``             |
+----------------------+----------------------------+

These are provided when constructing an :class:`~casadi_control.problem.ocp.OCP`.

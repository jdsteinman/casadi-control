User Guide
==========

CasADi-Control is built for continuous-time optimal control problems:
problems in which we seek a time-varying state and control trajectory
that minimize an objective while satisfying differential equations,
bounds, and path or endpoint constraints.

These problems appear throughout science and engineering. A few standard
examples are:

* trajectory design for aerospace vehicles and spacecraft
* motion planning and energy management in robotics
* process control in chemical and industrial systems
* estimation and inverse problems with dynamic models
* economics and resource-allocation problems with intertemporal dynamics


Problem Formulation
-------------------

CasADi-Control represents problems of the form

.. math::

   \begin{aligned}
   \min_{x(\cdot),u(\cdot),p}\quad
   & \Phi\bigl(x(t_0), x(t_f), p, t_0, t_f\bigr)
     + \int_{t_0}^{t_f} L\bigl(x(t), u(t), p, t\bigr)\,dt \\
   \text{s.t.}\quad
   & x'(t) = f\bigl(x(t), u(t), p, t\bigr), \\
   & c\bigl(x(t), u(t), p, t\bigr) \le 0, \\
   & s\bigl(x(t), p, t\bigr) \le 0, \\
   & b\bigl(x(t_0), x(t_f), p, t_0, t_f\bigr) = 0,
   \end{aligned}

where :math:`x(t)` is the state trajectory, :math:`u(t)` is the control,
and :math:`p` denotes constant decision parameters.

In the package, these objects are encoded by :class:`casadi_control.OCP`
callbacks:

.. code-block:: python

   from casadi_control import OCP

   ocp = OCP(
       n_x=...,
       n_u=...,
       f_dyn=f_dyn,
       l_run=l_run,
       l_end=l_end,
       path_constr=path_constr,
       state_constr=state_constr,
       bnd_constr=bnd_constr,
   )


Further Reading
---------------

Useful references for the mathematical background include standard texts
on optimal control, nonlinear programming, and direct transcription
methods. Good companion references typically include:

* classical texts on Pontryagin's principle and dynamic optimization,
  such as :cite:t:`bryson-ho1975`
* texts on nonlinear programming and KKT systems, such as
  :cite:t:`biegler2010`
* monographs on direct optimal control and transcription methods, such
  as :cite:t:`betts2010`
* the CasADi documentation for symbolic modeling and derivative-based
  NLP construction :cite:p:`casadi-docs`

The full bibliography is collected on the :doc:`references` page.

"""
Problem-definition layer for optimal control models.

This namespace provides the core objects used to define continuous-time
optimal control problems before discretization.

The represented problem class has the form

.. math::

    \\begin{aligned}
    \\min_{x,\\,u,\\,p}\\quad
    & \\phi\\big(x(t_0),x(t_f),p\\big)
      + \\int_{t_0}^{t_f} \\ell\\big(x(t),u(t),p,t\\big)\\,dt
    \\\\[6pt]
    \\text{s.t.}\\quad
    & x'(t) = f\\big(x(t),u(t),p,t\\big)
    & \\forall t \\in (t_0,t_f)
    \\\\
    &
    c\\big(x(t),u(t),p,t\\big) \\le 0
    & \\forall t \\in [t_0,t_f]
    \\\\
    &
    s\\big(x(t),p,t\\big) \\le 0
    & \\forall t \\in [t_0,t_f]
    \\\\
    &
    b\\big(x(t_0),x(t_f)\\big) = 0
    \\\\
    &
    u(t) \\in \\mathcal{U}
    & \\forall t \\in [t_0,t_f].
    \\end{aligned}

"""

from .ocp import OCP, Scaling


__all__ = ["OCP", "Scaling"]

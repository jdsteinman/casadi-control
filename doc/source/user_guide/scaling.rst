Scaling and Coordinate Systems
==============================

Optimal control problems often involve variables whose magnitudes differ
by several orders of magnitude. For example, a state vector may contain

* positions on the order of :math:`10^3`
* velocities on the order of :math:`10^2`
* angles on the order of :math:`10^{-1}`

Such disparities can degrade the numerical performance of nonlinear
optimization algorithms.

CasADi-Control therefore supports **automatic variable scaling**, which
transforms the optimal control problem into a numerically well-scaled
coordinate system before constructing the nonlinear program.

This chapter explains how scaling works and how it affects the problem
formulation.

Motivation
----------

Consider an optimal control problem with state

.. math::

    x(t) = \begin{bmatrix}
        y(t) \\
        v(t)
    \end{bmatrix}

where

* :math:`y` is altitude (thousands of meters)
* :math:`v` is velocity (hundreds of meters per second)

The solver must optimize over both variables simultaneously. If these
variables differ greatly in magnitude, the resulting nonlinear program
may have poor conditioning.

Scaling transforms the variables so they are closer to order unity.

Scaled Variables
----------------

Scaling introduces dimensionless variables

.. math::

    \hat{x} = \frac{x}{x_\mathrm{ref}}, \qquad
    \hat{u} = \frac{u}{u_\mathrm{ref}}, \qquad
    \hat{p} = \frac{p}{p_\mathrm{ref}}.

The reference values

.. math::

    x_\mathrm{ref},\; u_\mathrm{ref},\; p_\mathrm{ref}

are user-provided scaling factors representing typical magnitudes of the
corresponding variables.

The nonlinear program is then constructed using the scaled variables

.. math::

    \hat{x}, \hat{u}, \hat{p}.

Relationship Between Physical and Scaled Variables
--------------------------------------------------

The relationship between physical and scaled variables is

.. math::

    x = x_\mathrm{ref} \hat{x},
    \qquad
    u = u_\mathrm{ref} \hat{u},
    \qquad
    p = p_\mathrm{ref} \hat{p}.

User-provided model functions (dynamics, objective, constraints) are
always evaluated in **physical units**.

The library automatically converts between scaled and physical
variables when constructing the nonlinear program.

Time Scaling
------------

The time variable may also be scaled.

Define a dimensionless time coordinate

.. math::

    \hat{t} = \frac{t - t_0}{t_\mathrm{ref}}.

The relationship between derivatives in the two coordinate systems is

.. math::

    \frac{dx}{dt}
    =
    \frac{dx}{d\hat{t}}
    \frac{d\hat{t}}{dt}.

Since

.. math::

    t = t_0 + t_\mathrm{ref}\hat{t},

we obtain

.. math::

    \frac{dx}{d\hat{t}} = t_\mathrm{ref} \, f(x,u,p,t).

The discretization layer therefore uses the transformed dynamics

.. math::

    \hat{x}' = \frac{t_\mathrm{ref}}{x_\mathrm{ref}} f(x,u,p,t).

This transformation ensures that the scaled problem is equivalent to the
original physical problem.

Implementation in CasADi-Control
--------------------------------

Scaling is represented by the class

:class:`casadi_control.problem.scaling.Scaling`.

A scaling specification may include

* ``x_ref`` – state scaling
* ``u_ref`` – control scaling
* ``p_ref`` – parameter scaling
* ``t_ref`` – time scaling

The scaling transformation is applied using

.. code-block:: python

    scaled_ocp = ocp.scaled(scaling)

This produces a new :class:`~casadi_control.problem.ocp.OCP` instance
whose callbacks operate in the scaled coordinate system.

Internally, the transformation modifies

* dynamics
* objective functions
* constraints
* bounds
* initial guesses

so that the resulting problem is expressed entirely in scaled
coordinates.

Example
-------

Suppose the state vector contains

* altitude in meters
* velocity in meters per second

Typical magnitudes might be

.. code-block:: python

    scaling = Scaling(
        x_ref=[10000, 500],
        u_ref=[0.1],
    )

Applying the transformation

.. code-block:: python

    ocp_scaled = ocp.scaled(scaling)

produces an equivalent optimal control problem whose variables are of
order unity.

Best Practices
--------------

Choose scaling factors so that

* state variables are typically between 0.1 and 10
* control variables are typically between 0.1 and 10
* parameters are order unity

This improves the conditioning of the nonlinear program and typically
reduces the number of solver iterations.

Scaling should reflect **typical magnitudes**, not strict bounds.

For example, if altitude ranges from 0 to 20 000 meters, a good scaling
might be

.. math::

    x_\mathrm{ref} = 10\,000.

rather than 20 000.

Free Final Time
---------------

If the final time is free, scaling also affects the optimization
variable representing the final time.

If

.. math::

    t_f

is a decision variable, the scaled problem optimizes

.. math::

    \hat{t}_f = \frac{t_f}{t_\mathrm{ref}}.

The discretization layer automatically applies the appropriate
transformations so that the physical objective and constraints remain
unchanged.

Design Philosophy
-----------------

CasADi-Control treats scaling as a **coordinate transformation of the
optimal control problem**.

This approach has several advantages:

* the continuous problem definition remains in physical units
* the solver operates in well-scaled variables
* transformations are applied systematically and consistently

This design avoids the need for manual scaling within user-provided
model functions.

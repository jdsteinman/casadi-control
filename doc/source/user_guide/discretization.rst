Discretization
==============

Continuous-time optimal control problems must be converted into finite
dimensional nonlinear programs before they can be solved numerically.

CasADi-Control performs this conversion using **direct transcription
methods**, which transform the optimal control problem into a nonlinear
program (NLP) by discretizing the state and control trajectories.

This chapter explains how the discretization layer of the library is
organized and how continuous optimal control problems are converted into
finite-dimensional optimization problems.

Overview
--------

The typical pipeline for solving an optimal control problem is

.. code-block::

    OCP
     ↓
    Discretization
     ↓
    Nonlinear Program (NLP)
     ↓
    NLP Solver
     ↓
    Solution / Postprocessing

The library separates these stages into distinct components:

+------------------+-----------------------------------+
| Component        | Responsibility                    |
+==================+===================================+
| ``problem``      | Continuous-time OCP definition    |
+------------------+-----------------------------------+
| ``discretization`` | Transcription into an NLP       |
+------------------+-----------------------------------+
| ``solvers``      | Numerical NLP solution            |
+------------------+-----------------------------------+
| ``io``           | Storage of artifacts and results  |
+------------------+-----------------------------------+

The discretization layer is responsible for converting a continuous
optimal control problem into the algebraic constraints required by a
nonlinear program.

Direct Transcription
--------------------

CasADi-Control implements **direct transcription methods**.

In direct transcription, the continuous trajectories are approximated
using a finite set of decision variables defined on a time grid.

For example, the state trajectory

.. math::

    x(t)

is approximated by values

.. math::

    x_0, x_1, \dots, x_N

defined at discrete nodes.

The dynamics are enforced through algebraic constraints that ensure the
trajectory satisfies the differential equation approximately.

These constraints are called **defect constraints**.

Direct Collocation
------------------

The current implementation uses **direct collocation** methods.

In direct collocation, the state trajectory within each time interval is
approximated by a polynomial. The dynamics are enforced by requiring the
polynomial derivative to match the differential equation at selected
collocation points.

This produces a set of algebraic constraints of the form

.. math::

    \dot{x}_k - f(x_k, u_k, p, t_k) = 0.

The resulting nonlinear program contains

* state variables
* control variables
* parameters (optional)
* defect constraints
* path constraints
* boundary constraints

The primary entry point for this transcription is

:class:`casadi_control.discretization.collocation.DirectCollocation`.

Discretization Architecture
---------------------------

The discretization module is organized around a set of base interfaces
and specific transcription implementations.

The structure of the module is roughly

.. code-block::

    discretization
    ├── base
    │   ├── Transcription
    │   └── DiscreteSolution
    │
    ├── collocation
    │   ├── DirectCollocation
    │   ├── trajectory
    │   ├── postprocess
    │   └── schemes
    │
    ├── rk
    └── theta

The base module defines the abstract interfaces used by all
transcription methods. Specific methods (such as collocation) implement
these interfaces.

Base Interfaces
---------------

The base classes define the general structure required for a
discretization method.

Key responsibilities include

* constructing decision variables
* assembling defect constraints
* assembling path constraints
* assembling boundary constraints
* constructing the final NLP

Different transcription methods implement these operations using
different discretization schemes.

Time Grids
----------

The transcription process introduces a finite time grid

.. math::

    t_0 < t_1 < \dots < t_N.

Decision variables are introduced at these nodes.

Additional intermediate nodes may also be introduced depending on the
collocation scheme.

For example, Legendre–Gauss–Radau collocation introduces additional
internal nodes within each interval.

Decision Variables
------------------

The nonlinear program typically contains the following decision
variables:

State variables
    Values of the state trajectory at discretization nodes.

Control variables
    Values of the control input at discretization nodes.

Parameters
    Constant decision variables included in the optimal control
    problem.

Final time
    If the problem has free final time.

The discretization layer is responsible for constructing the vector of
decision variables used by the NLP solver.

Constraint Assembly
-------------------

Constraints in the NLP originate from several sources:

Dynamics constraints
    Enforce the differential equation.

Path constraints
    Enforce inequalities along the trajectory.

State constraints
    Constraints depending only on the state.

Boundary constraints
    Constraints linking the initial and final state.

These constraints are assembled into a single vector

.. math::

    g(z)

where :math:`z` is the vector of decision variables.

The nonlinear program takes the form

.. math::

    \min_z \; J(z)

subject to

.. math::

    g(z) = 0,
    \qquad
    g(z) \le 0.

Postprocessing
--------------

After solving the nonlinear program, the raw solver output must be
converted back into trajectory form.

The discretization module therefore provides postprocessing utilities
that

* reconstruct trajectories
* compute derived quantities
* expose convenient accessors for states and controls.

These operations are implemented in the ``postprocess`` and
``trajectory`` modules.

Future Transcription Methods
----------------------------

The architecture of the discretization module allows additional
transcription methods to be implemented.

Possible future implementations include

* Runge–Kutta transcription
* Multiple shooting
* Trapezoidal or theta methods

Each new method implements the same base interfaces and can therefore
integrate with the rest of the library without modifying the solver
layer.

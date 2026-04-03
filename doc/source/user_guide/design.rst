Design Philosophy
=================

CasADi-Control is designed to provide a clear and modular framework for
formulating and solving optimal control problems using CasADi.

The architecture separates the components of an optimal control solver
into independent layers. This separation improves clarity, extensibility,
and testability.

This chapter explains the guiding design principles of the library.

Layered Architecture
--------------------

The library separates the optimal control workflow into several layers:

.. code-block::

    Continuous problem definition
           │
           ▼
    Discretization / transcription
           │
           ▼
    Nonlinear program (NLP)
           │
           ▼
    Numerical solver
           │
           ▼
    Postprocessing and analysis

Each layer has a distinct responsibility.

Problem Definition
~~~~~~~~~~~~~~~~~~

The :mod:`casadi_control.problem` module defines the continuous-time
optimal control problem.

This layer contains

* system dynamics
* objective functions
* constraints
* bounds

The problem definition is independent of any discretization method or
numerical solver.

Discretization
~~~~~~~~~~~~~~

The :mod:`casadi_control.discretization` module converts the continuous
optimal control problem into a nonlinear program.

Different transcription methods can be implemented within this layer,
including

* direct collocation
* Runge–Kutta transcription
* multiple shooting

Because discretization is separated from the problem definition, the
same optimal control problem can be solved using different transcription
methods.

Solvers
~~~~~~~

The :mod:`casadi_control.solvers` module interfaces with nonlinear
programming solvers.

Typical solvers include

* IPOPT
* SNOPT
* other NLP solvers supported by CasADi

The solver layer operates only on the nonlinear program produced by the
discretization layer.

Postprocessing
~~~~~~~~~~~~~~

After solving the nonlinear program, the solution vector must be
converted into trajectory data.

The postprocessing layer reconstructs

* state trajectories
* control trajectories
* auxiliary quantities

from the raw solver output.

Separation of Physical and Solver Coordinates
---------------------------------------------

The library distinguishes between two coordinate systems:

Physical coordinates
    The variables used in the user-defined model functions.

Scaled coordinates
    The variables used internally by the nonlinear solver.

The :class:`casadi_control.problem.scaling.Scaling` class defines the
transformation between these coordinate systems.

User-provided callbacks always operate in **physical units**, while the
solver may operate in scaled coordinates for numerical stability.

Callback-Based Problem Definition
---------------------------------

Optimal control problems are defined using Python callables.

For example, the system dynamics are defined as

.. code-block:: python

    def f_dyn(x, u, p, t):
        ...

This approach provides several advantages:

* full flexibility in defining models
* compatibility with CasADi symbolic expressions
* easy integration with existing model code

Callbacks may return CasADi expressions or numeric values.

Vector Convention
-----------------

Vectors in CasADi-Control follow the convention

.. math::

    x \in \mathbb{R}^{n_x \times 1}.

This convention matches CasADi's internal representation of symbolic
vectors and avoids ambiguity between row and column vectors.

Constraints returned by callbacks should therefore have shape

.. math::

    (m, 1).

Empty constraints may be represented using empty vectors.

Immutability of Problem Objects
-------------------------------

The :class:`casadi_control.problem.ocp.OCP` class is implemented as an
immutable dataclass.

Once created, the structure of an optimal control problem does not
change.

This design has several benefits:

* problems can be reused safely
* transformations (such as scaling) produce new problem objects
* accidental mutation is avoided

For example, applying scaling produces a new problem instance

.. code-block:: python

    ocp_scaled = ocp.scaled(scaling)

while leaving the original problem unchanged.

Explicit Transcription Objects
------------------------------

Discretization methods are implemented as explicit objects rather than
hidden behind helper functions.

For example

.. code-block:: python

    transcription = DirectCollocation(ocp, N=100)

This design allows the transcription object to expose additional
functionality such as

* inspection of the NLP
* warm-start initialization
* intermediate diagnostics

Extensibility
-------------

The architecture of the library is designed to support new components
without modifying existing code.

Examples include

* new transcription methods
* alternative NLP solvers
* new collocation schemes

Because the modules interact through well-defined interfaces, these
extensions can be added incrementally.

Consistency with CasADi
-----------------------

CasADi-Control follows conventions used throughout CasADi.

These include

* symbolic expressions represented by :class:`casadi.MX`
* vector shapes ``(n,1)``
* solver interfaces based on CasADi NLP functions

This consistency allows users already familiar with CasADi to work
efficiently with the library.

Long-Term Goals
---------------

The long-term goal of CasADi-Control is to provide a flexible research
platform for optimal control algorithms.

The design therefore prioritizes

* modular architecture
* mathematical transparency
* extensibility

over minimizing the number of exposed abstractions.

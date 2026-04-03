:html_theme.sidebar_secondary.remove: true

*****************
CasADi-Control
*****************

**Date**: |today| **Version**: |version|

**Useful links**:
`Source Repository <https://github.com/YOUR_ORG_OR_YOU/casadi-control>`__ |
`Issues <https://github.com/YOUR_ORG_OR_YOU/casadi-control/issues>`__

CasADi-Control is a Python library for formulating and solving continuous-time
optimal control problems using CasADi-based nonlinear programming.

The library provides:

* A structured representation of optimal control problems
* Direct transcription methods (collocation)
* Solver interfaces
* Artifact storage and warm-start support
* Tools for postprocessing and visualization


.. grid:: 1 1 2 2
   :gutter: 2 3 4 4

   .. grid-item-card::
      :img-top: _static/cd.svg
      :text-align: center

      **Installation**
      ^^^

      Install CasADi-Control.

      +++

      .. button-ref:: installation
         :expand:
         :color: secondary
         :click-parent:

         To installation

   .. grid-item-card::
      :img-top: _static/user_guide.svg
      :text-align: center

      **User guide**
      ^^^

      In-depth explanations of problem setup, discretization, scaling, solvers,
      and postprocessing.

      +++

      .. button-ref:: user_guide/index
         :expand:
         :color: secondary
         :click-parent:

         To the user guide

   .. grid-item-card::
      :img-top: _static/api.svg
      :text-align: center

      **API reference**
      ^^^

      Complete API documentation for the public classes and functions.

      +++

      .. button-ref:: reference/index
         :expand:
         :color: secondary
         :click-parent:

         To the API reference

   .. grid-item-card::
      :img-top: _static/example.svg
      :text-align: center

      **Examples**
      ^^^

      Worked examples and recipes you can adapt for your own optimal control problems.

      +++

      .. button-ref:: examples/index
         :expand:
         :color: secondary
         :click-parent:

         To the examples


.. toctree::
   :maxdepth: 1
   :hidden:

   Installation <installation>
   User guide <user_guide/index>
   Examples <examples/index>
   API reference <reference/index>

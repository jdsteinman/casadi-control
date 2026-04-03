Installation
============

Requirements
------------

CasADi-Control requires:

* Python ≥ 3.10
* ``pip`` or another PEP 517-compatible installer

Base installation
-----------------

From the repository root, install the package with:

.. code-block:: bash

   pip install .

For local development, prefer an editable install:

.. code-block:: bash

   pip install -e .

To confirm the package imports successfully:

.. code-block:: bash

   python -c "import casadi_control; print(casadi_control.__all__[:3])"

Optional dependency groups
--------------------------

Install extras depending on your workflow:

.. code-block:: bash

   # tests
   pip install -e ".[test]"

   # plotting examples
   pip install -e ".[examples]"

   # Sphinx docs
   pip install -e ".[docs]"

   # full development setup
   pip install -e ".[dev]"

These extras are additive. For example, ``.[dev]`` is the easiest choice if
you want the test, documentation, and example dependencies in one environment.

Development shortcuts
---------------------

The repository includes a top-level ``Makefile`` with a few common tasks:

.. code-block:: bash

   make install-dev
   make test
   make docs

You can also run examples directly from the repository root:

.. code-block:: bash

   python examples/hager_lq_ocp.py
   python examples/hager_hou_rao_ocp.py

Build documentation
-------------------

Install the docs dependencies and build HTML:

.. code-block:: bash

   pip install -e ".[docs]"
   make -C doc html

Open:

.. code-block:: text

   doc/build/html/index.html

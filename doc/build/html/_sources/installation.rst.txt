Installation
============

Requirements
------------

CasADi-Control requires:

* Python ≥ 3.10
* pip

Base installation
-----------------

Install the package:

.. code-block:: bash

   pip install .

For local development (editable install):

.. code-block:: bash

   pip install -e .

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

Build documentation
-------------------

Install the docs dependencies and build HTML:

.. code-block:: bash

   pip install -e ".[docs]"
   make -C doc html

Open:

.. code-block:: text

   doc/build/html/index.html

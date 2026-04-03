CONDA_ENV ?= casadi-ocp
NOTEBOOK_EXEC_ENV = env PYTHONPATH=$(CURDIR)/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl IPYTHONDIR=/tmp/ipython
JUPYTEXT_EXEC = $(NOTEBOOK_EXEC_ENV) conda run -n $(CONDA_ENV) python -m jupytext --to ipynb --execute

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -q

docs:
	make -C doc html

docs-pages:
	NBSPHINX_EXECUTE=never sphinx-build -b html doc/source doc/build/html

sync-examples:
	python -m jupytext --sync examples/hager_lq_ocp.py examples/hager_hou_rao_ocp.py

sync-example-lq:
	python -m jupytext --sync examples/hager_lq_ocp.py

sync-example-hhr:
	python -m jupytext --sync examples/hager_hou_rao_ocp.py

refresh-notebooks:
	$(JUPYTEXT_EXEC) examples/hager_lq_ocp.py -o examples/hager_lq_ocp.ipynb
	$(JUPYTEXT_EXEC) examples/hager_hou_rao_ocp.py -o examples/hager_hou_rao_ocp.ipynb

refresh-notebook-lq:
	$(JUPYTEXT_EXEC) examples/hager_lq_ocp.py -o examples/hager_lq_ocp.ipynb

refresh-notebook-hhr:
	$(JUPYTEXT_EXEC) examples/hager_hou_rao_ocp.py -o examples/hager_hou_rao_ocp.ipynb

example-lq:
	python examples/hager_lq_ocp.py

example-hhr:
	python examples/hager_hou_rao_ocp.py

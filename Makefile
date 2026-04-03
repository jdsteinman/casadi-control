install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -q

docs:
	make -C doc html

sync-examples:
	python -m jupytext --sync examples/hager_lq_ocp.py examples/hager_hou_rao_ocp.py

sync-example-lq:
	python -m jupytext --sync examples/hager_lq_ocp.py

sync-example-hhr:
	python -m jupytext --sync examples/hager_hou_rao_ocp.py

example-lq:
	python examples/hager_lq_ocp.py

example-hhr:
	python examples/hager_hou_rao_ocp.py

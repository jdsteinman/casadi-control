install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -q

docs:
	make -C doc html

example-lq:
	python examples/hager_lq_ocp.py

example-hhr:
	python examples/hager_hao_rao_ocp.py

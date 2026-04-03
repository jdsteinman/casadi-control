# casadi-control

Python tools for formulating and solving continuous-time optimal control
problems with CasADi.

## Requirements

- Python 3.10+
- `pip` (or another PEP 517-compatible installer)

## Installation

Install the package:

```bash
pip install .
```

For local development (editable install):

```bash
pip install -e .
```

Verify the install:

```bash
python -c "import casadi_control; print('casadi_control imported successfully')"
```

## Optional dependency groups

Install extras based on what you need:

- Tests:
  ```bash
  pip install -e ".[test]"
  ```
- Examples (plotting):
  ```bash
  pip install -e ".[examples]"
  ```
- Documentation:
  ```bash
  pip install -e ".[docs]"
  ```
- Full development setup:
  ```bash
  pip install -e ".[dev]"
  ```

## Running tests

```bash
pytest -q
```

## Running examples

Run scripts in `examples/`, for example:

```bash
python examples/hager_lq_ocp.py
python examples/hager_hao_rao_ocp.py
```

## Building documentation

From the repository root:

```bash
pip install -e ".[docs]"
make -C doc html
```

Built docs will be in `doc/build/html/index.html`.


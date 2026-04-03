# casadi-control

`casadi-control` is a Python library for formulating, discretizing, and
solving continuous-time optimal control problems with
[CasADi](https://web.casadi.org/).

It provides a compact workflow for:

- defining optimal control problems with clear callback-based APIs
- transcribing those problems with direct collocation
- solving the resulting nonlinear programs with IPOPT
- postprocessing primal and dual trajectories for analysis and plotting

The project includes worked examples, Sphinx documentation, and a small
high-level solve API for standard end-to-end workflows.

## Requirements

- Python 3.10 or newer
- `pip` or another PEP 517-compatible installer

## Installation

Clone the repository and install the package from the project root:

```bash
pip install .
```

For local development, use an editable install:

```bash
pip install -e .
```

To verify the installation:

```bash
python -c "import casadi_control; print(casadi_control.__all__[:3])"
```

### Optional dependency groups

Install the extra dependencies that match your workflow:

- Test suite:
  ```bash
  pip install -e ".[test]"
  ```
- Examples and plotting:
  ```bash
  pip install -e ".[examples]"
  ```
- Documentation:
  ```bash
  pip install -e ".[docs]"
  ```
- Full development environment:
  ```bash
  pip install -e ".[dev]"
  ```

## Quick start

The package centers around three concepts:

1. `OCP` for defining a continuous-time optimal control problem
2. a discretization such as `DirectCollocation`
3. a solver step, typically via `solve(...)` or `solve_ipopt(...)`

```python
from casadi_control import OCP, DirectCollocation, solve

# Define the problem callbacks and dimensions in OCP(...)
ocp = OCP(...)

tx = DirectCollocation(N=40, degree=3, scheme="flgr")
result = solve(ocp, tx)

sol = result.sol
pp = result.pp
```

For a complete example, start with
[`examples/hager_lq_ocp.py`](examples/hager_lq_ocp.py).

## Examples

The repository includes worked examples in [`examples/`](examples/):

- `hager_lq_ocp.py` solves a linear-quadratic benchmark and compares the
  numerical solution against the analytical one.
- `hager_hou_rao_ocp.py` demonstrates a more general benchmark problem.

Run them directly from the repository root:

```bash
python examples/hager_lq_ocp.py
python examples/hager_hou_rao_ocp.py
```

The example notebooks are tracked as Jupytext pairs. Edit the `.py` source,
then synchronize the notebooks with:

```bash
make sync-examples
```

Or sync individual examples:

```bash
make sync-example-lq
make sync-example-hhr
```

## Development

Common project tasks are exposed through the top-level `Makefile`:

```bash
make install-dev
make test
make docs
```

You can also run the test suite directly:

```bash
pytest -q
```

## Documentation

Build the Sphinx documentation from the repository root with:

```bash
pip install -e ".[docs]"
make -C doc html
```

The built site will be available at `doc/build/html/index.html`.

## Project links

- Repository: <https://github.com/jdsteinman/casadi-control>
- Issue tracker: <https://github.com/jdsteinman/casadi-control/issues>

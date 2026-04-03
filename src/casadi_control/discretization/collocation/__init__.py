"""Collocation discretization public API.

Stable surface:

- :class:`~casadi_control.discretization.collocation.DirectCollocation`
- :func:`~casadi_control.discretization.collocation.save_npz`
- :func:`~casadi_control.discretization.collocation.load_npz`

Other modules in this package (initialization helpers, archive utilities,
postprocessing views, and scheme internals) are implementation details and may
change without notice.
"""

from __future__ import annotations

from .archive import load_npz, save_npz
from .direct_collocation import DirectCollocation

__all__ = [
    "DirectCollocation",
    "save_npz",
    "load_npz",
]

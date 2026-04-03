"""Collocation discretization (public API).

Only :class:`~casadi_control.discretization.collocation.DirectCollocation` is
considered public and stable.

Other modules in this package (initialization helpers, archive utilities, views,
and scheme internals) are implementation details and may change without notice.
"""

from __future__ import annotations

from .direct_collocation import DirectCollocation

# ---------------------------------------------------------------------------
# Internal / legacy helpers
# ---------------------------------------------------------------------------
# These are intentionally not exported via __all__. They may be removed or
# changed in future versions without deprecation.
from .archive import save_npz, load_npz
from .schemes import make_table as _make_table  # noqa: F401

__all__ = [
    "DirectCollocation",
    "save_npz",
    "load_npz"
]

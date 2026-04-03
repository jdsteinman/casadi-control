"""Collocation table factories.

This module selects and instantiates collocation coefficient tables used by
the direct-collocation transcription.
"""

from .base import CollocationTable
from .flgr import flgr_table


def make_table(name: str, degree: int) -> CollocationTable:
    """Create a collocation coefficient table.

    Parameters
    ----------
    name : str
        Scheme identifier. Currently supported aliases are ``"flgr"`` and
        ``"radau"``.
    degree : int
        Number of collocation points per interval.

    Returns
    -------
    CollocationTable
        Coefficient table for the selected scheme and degree.

    Raises
    ------
    ValueError
        If ``name`` is not recognized.
    """
    name = name.lower()
    if name in ("flgr", "radau"):
        return flgr_table(degree)
    raise ValueError(f"Unknown collocation scheme '{name}'")

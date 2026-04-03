"""Discretization registry and factory helpers."""

from __future__ import annotations

from typing import Callable, Dict, Iterable

from .base import Discretization
from .collocation import DirectCollocation


def _normalize_name(name: str) -> str:
    """Normalize a discretization key.

    Parameters
    ----------
    name : str
        Discretization identifier.

    Returns
    -------
    str
        Normalized, lowercased key suitable for registry lookup.
    """
    key = str(name).strip().lower()
    if not key:
        raise ValueError("discretization name must be non-empty")
    return key


class DiscretizationRegistry:
    """Name-to-factory mapping for discretization constructors.

    This registry maps string names (e.g. ``"collocation"``) to callables that
    construct :class:`~casadi_control.discretization.base.Discretization` objects.

    Notes
    -----
    The default registry is used by :func:`DiscretizationFactory`.
    """

    def __init__(
        self,
        entries: Iterable[tuple[str, Callable[..., Discretization]]] | None = None,
    ) -> None:
        self._factories: Dict[str, Callable[..., Discretization]] = {}
        if entries is not None:
            for name, factory in entries:
                self.register(name, factory)

    def register(
        self,
        name: str,
        factory: Callable[..., Discretization],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a discretization constructor under ``name``."""
        key = _normalize_name(name)
        if not callable(factory):
            raise TypeError("factory must be callable")
        if key in self._factories and not overwrite:
            raise ValueError(
                f"discretization '{key}' is already registered; use overwrite=True to replace it"
            )
        self._factories[key] = factory

    def create(self, name: str, /, *args, **kwargs) -> Discretization:
        """Instantiate a discretization by ``name``."""
        key = _normalize_name(name)
        try:
            factory = self._factories[key]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise KeyError(
                f"Unknown discretization '{name}'. Registered: {available}"
            ) from exc
        return factory(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        """Return registered discretization names."""
        return tuple(sorted(self._factories))


_DEFAULT_REGISTRY = DiscretizationRegistry(
    entries=(
        ("direct_collocation", DirectCollocation),
        ("collocation", DirectCollocation),
    )
)


def DiscretizationFactory(name: str, /, *args, **kwargs) -> Discretization:
    """Construct a discretization by name from the default registry."""
    return _DEFAULT_REGISTRY.create(name, *args, **kwargs)


def available_discretizations() -> tuple[str, ...]:
    """Return names available in the default discretization registry."""
    return _DEFAULT_REGISTRY.names()


__all__ = [
    "DiscretizationRegistry",
    "DiscretizationFactory",
    "available_discretizations",
]

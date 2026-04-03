import pytest

from casadi_control import (
    DiscretizationFactory,
    available_discretizations,
)
from casadi_control.discretization import Discretization, DirectCollocation, DiscretizationRegistry


class _DummyDiscretization(Discretization):
    name = "dummy"

    def build(self, ocp):
        raise NotImplementedError

    def postprocess(self, ocp, nlp, sol):
        raise NotImplementedError

    def to_artifact(self, sol, pp):
        raise NotImplementedError

    def from_artifact(self, art):
        raise NotImplementedError


def test_default_registry_exposes_collocation_aliases():
    names = available_discretizations()
    assert "collocation" in names
    assert "direct_collocation" in names


def test_discretization_factory_from_default_registry():
    disc = DiscretizationFactory("collocation", N=5, degree=3)
    assert isinstance(disc, DirectCollocation)


def test_discretization_factory_alias():
    disc = DiscretizationFactory("direct_collocation", N=5, degree=3)
    assert isinstance(disc, DirectCollocation)


def test_registry_duplicate_and_overwrite_behavior():
    reg = DiscretizationRegistry()
    reg.register("dummy", _DummyDiscretization)
    with pytest.raises(ValueError):
        reg.register("dummy", DirectCollocation)

    reg.register("dummy", DirectCollocation, overwrite=True)
    disc = reg.create("dummy", N=4)
    assert isinstance(disc, DirectCollocation)


def test_registry_unknown_name_error_lists_registered_names():
    reg = DiscretizationRegistry(entries=(("dummy", _DummyDiscretization),))
    with pytest.raises(KeyError, match="Registered: dummy"):
        reg.create("missing")

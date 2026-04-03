import numpy as np

from casadi_control.discretization.collocation import collocation_coefficients


def test_collocation_coefficients_legacy_shapes():
    tau, b, c, D = collocation_coefficients(3)

    assert np.asarray(tau).shape == (4,)
    assert np.asarray(b).shape == (4,)
    assert np.asarray(c).shape == (4,)
    assert np.asarray(D).shape == (3, 4)

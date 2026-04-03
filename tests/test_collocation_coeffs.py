import numpy as np
from scipy.special import legendre
from scipy.interpolate import lagrange

from casadi_control.discretization.collocation import collocation_coefficients


# --- Reference implementation ---
def radau_coeffs(K):
    deltas = [-1.0 / ((2.0 * p + 1.0) * (2.0 * p + 3.0)) for p in range(K)]
    gammas = [
        np.sqrt(
            4.0 * (p + 2.0) ** 2.0 * (p + 1.0) ** 2.0
            / ((2.0 * p + 3.0) ** 2.0 * (2.0 * p + 4.0) * (2.0 * p + 2.0))
        )
        for p in range(K - 1)
    ]
    return np.array(deltas), np.array(gammas)


def flgr_eigenvalues(K):
    M = np.zeros((K - 1, K - 1))
    deltas, gammas = radau_coeffs(K - 1)
    for n in range(deltas.shape[0]):
        M[n, n] = deltas[n]
    for n in range(gammas.shape[0]):
        M[n + 1, n] = gammas[n]
        M[n, n + 1] = gammas[n]

    tau = np.append(np.sort(np.linalg.eigvals(M)), [1])
    return tau


def FLGR_info(K, a=0.0, b=1.0):
    if K == 3:
        tau_ref = 2.0 * np.array(
            [2.0 / 5.0 - np.sqrt(6.0) / 10, 2.0 / 5.0 + np.sqrt(6.0) / 10, 1.0]
        ) - 1.0
        weights = (b - a) * np.array(
            [4.0 / 9.0 - np.sqrt(6.0) / 36.0, 4.0 / 9.0 + np.sqrt(6.0) / 36.0, 1.0 / 9.0]
        )
    else:
        tau_ref = flgr_eigenvalues(K)
        weights = (
            (b - a)
            / 2.0
            * (1.0 + tau_ref)
            / K**2
            * (1.0 / np.polyval(legendre(K - 1), tau_ref)) ** 2.0
        )

    # interpolation points on [-1,1]
    intpoints = np.append([-1.0], tau_ref)

    # derivatives of Lagrange basis
    dL = []
    for i in range(K + 1):
        y = np.zeros(K + 1)
        y[i] = 1.0
        p = lagrange(intpoints, y)
        dp = np.polyder(p)
        dL.append(dp)

    # differentiation matrix on (a,b]
    D = np.zeros((K, K + 1))
    for k in range(K):
        for j in range(K + 1):
            D[k, j] = 2.0 / (b - a) * np.polyval(dL[j], tau_ref[k])

    # map nodes to (a,b]
    tau = (b - a) * (tau_ref + 1.0) / 2.0 + a
    return tau, weights, D

# --- Test ---
def test_nodes_weights_D_match_flgr():
    for K in [2, 3, 4, 5]:
        tau_ref, w_ref, D_ref = FLGR_info(K, a=0.0, b=1.0)

        tau, b, c, D = collocation_coefficients(K)
        tau = np.asarray(tau, float)
        b = np.asarray(b, float)
        D = np.asarray(D, float)

        print(D_ref)
        print(D)

        assert np.allclose(tau[1:], tau_ref, atol=1e-12, rtol=0.0)

        assert np.allclose(b[1:], w_ref, atol=1e-12, rtol=0.0)

        assert np.allclose(D, D_ref, atol=1e-10, rtol=0.0)

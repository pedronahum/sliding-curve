"""The quadrature-based formulas in gaussian.py against the paper's printed
closed forms: flat curve (93), (94), (96) and the kappa = 0 case (D.5)."""
import numpy as np
import pytest

from slidingcurve import Curve, gaussian as g

K, D, S, R0 = 0.03, 5.0, 0.01, 0.04
Bp = lambda s: (1.0 - np.exp(-K * s)) / K        # the paper's B, kappa > 0


@pytest.mark.parametrize("T", [1.0, 10.0, 40.0])
def test_convexity_eq94(T):
    paper = S**2 / (2 * T * K**2) * (T - 2 * Bp(T) + Bp(2 * T) / 2)
    np.testing.assert_allclose(g.convexity(K, S, T), paper, rtol=1e-9)


@pytest.mark.parametrize("T", [1.0, 10.0, 40.0])
def test_flat_curve_eq93_eq96(T):
    c = Curve.flat(R0)
    roll = Bp(D) * S**2 / (4 * K * T) * (T - Bp(2 * T) / 2)
    np.testing.assert_allclose(g.yield_eh(c, K, D, S, T), R0 - roll, rtol=1e-10)
    np.testing.assert_allclose(g.term_premium(c, K, D, S, T),
                               roll + g.convexity(K, S, T), rtol=1e-9)


@pytest.mark.parametrize("T", [1.0, 10.0, 40.0])
def test_kappa_zero_d5(T):
    c = Curve.flat(R0)
    np.testing.assert_allclose(g.convexity(0.0, S, T), S**2 * T**2 / 6, rtol=1e-10)
    np.testing.assert_allclose(g.yield_eh(c, 0.0, D, S, T), R0 - S**2 * D * T / 4, rtol=1e-10)
    # small kappa is continuous with kappa = 0
    np.testing.assert_allclose(g.convexity(1e-9, S, T), S**2 * T**2 / 6, rtol=1e-6)


def test_forward_measure_mean_vanishes_at_maturity():
    # Remark 15: I(tau, tau) = 0
    assert abs(float(g.mean_x_fwd(K, S, 12.0, 12.0))) < 1e-18

import numpy as np
import jax.numpy as jnp

from slidingcurve import Curve


def test_flat_curve():
    c = Curve.flat(0.03)
    T = jnp.array([0.0, 1.0, 30.0])
    np.testing.assert_allclose(c.forward(T), 0.03, atol=1e-15)
    np.testing.assert_allclose(c.fwd_yield(T, T + 5.0), 0.03, atol=1e-15)
    np.testing.assert_allclose(c.discount(10.0), np.exp(-0.3), rtol=1e-14)


def test_nelson_siegel_forward_matches_analytic(ns_curve):
    # NS instantaneous forward: b0 + b1 e^{-u} + b2 u e^{-u}, u = T / tau
    T = jnp.linspace(0.0, 40.0, 81)
    u = T / 4.0
    f = 0.045 - 0.015 * jnp.exp(-u) + 0.01 * u * jnp.exp(-u)
    np.testing.assert_allclose(ns_curve.forward(T), f, atol=1e-13)


def test_forward_yield_is_average_forward(ns_curve):
    u = jnp.linspace(7.0, 12.0, 20001)
    avg = jnp.trapezoid(ns_curve.forward(u), u) / 5.0
    np.testing.assert_allclose(ns_curve.fwd_yield(7.0, 12.0), avg, rtol=1e-8)


def test_spline_curve_reprices_nodes():
    times = np.array([0.5, 1, 2, 5, 10, 20, 30.0])
    zeros = np.array([0.040, 0.039, 0.037, 0.038, 0.041, 0.044, 0.043])
    c = Curve.from_zero_nodes(times, zeros)
    np.testing.assert_allclose(c.zero(times), zeros, atol=1e-14)
    assert np.all(np.isfinite(c.forward(jnp.linspace(0.0, 40.0, 50))))

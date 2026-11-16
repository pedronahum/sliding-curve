"""Swaption pricer against the exact Gaussian price (Jamshidian), and the
forward-measure bond martingale away from the Gaussian case."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from slidingcurve import bachelier as bl, gaussian as g, pricing as pr, volscale

KEY = jax.random.PRNGKey(7)


def test_bachelier_round_trip():
    k = np.array([0.02, 0.04, 0.06])
    p = np.asarray(bl.normal_price(0.04, k, 0.0085, 10.0))
    np.testing.assert_allclose(bl.implied_normal_vol(p, 0.04, k, 10.0), 0.0085, rtol=1e-8)


def test_displaced_black_limits():
    # huge displacement with vol scaled down is Bachelier
    d = 1e4
    np.testing.assert_allclose(bl.displaced_black_price(0.04, 0.045, 0.01 / d, 5.0, d),
                               bl.normal_price(0.04, 0.045, 0.01, 5.0), rtol=1e-4)


@pytest.mark.parametrize("swp", [pr.Swaption.regular(10.0, 5.0), pr.Swaption.caplet(20.0, 5.0)],
                         ids=["10x5 swaption", "20y caplet on 5y rate"])
def test_gaussian_swaption_matches_jamshidian(ns_curve, base_model, swp):
    m = base_model
    fwd = float(pr.forward_swap_rate(ns_curve, swp))
    strikes = fwd + np.array([-0.01, 0.0, 0.01])
    ybar = lambda t: volscale.ybar_zeroth(ns_curve, m.delta, t)
    p, se = pr.swaption_prices(m, ns_curve, swp, strikes, ybar, KEY, n_paths=2**15)
    exact = np.array([g.payer_swaption(ns_curve, m.kappa, m.sigma, swp.expiry,
                                       swp.pay_times, swp.accruals, k) for k in strikes])
    assert np.all(np.abs(np.asarray(p) - exact) < 4 * np.asarray(se))
    np.testing.assert_allclose(p, exact, rtol=0.02)


@pytest.mark.parametrize("b,eta", [(0.0, 0.0), (0.6, 0.0), (0.0, 0.8), (0.6, 0.8)])
def test_forward_measure_bond_martingale(ns_curve, base_model, b, eta):
    """E^T[P(T, M)] = P(0, M) / P(0, T) for any volatility specification."""
    m = base_model._replace(b=b, eta=eta)
    ybar = lambda t: volscale.ybar_zeroth(ns_curve, m.delta, t)
    T, M = 10.0, jnp.array([12.0, 15.0, 20.0])
    state = pr.simulate_terminal(m, ns_curve, T, ybar, KEY, n_paths=2**15)
    x, y = state.x, state.y
    p = pr.bond(ns_curve, m.kappa, T, M[:, None], x[None, :], y[None, :])
    pairs = 0.5 * (p[:, : p.shape[1] // 2] + p[:, p.shape[1] // 2:])
    se = np.std(pairs, axis=1, ddof=1) / np.sqrt(pairs.shape[1])
    target = np.asarray(ns_curve.discount(M) / ns_curve.discount(T))
    assert np.all(np.abs(np.mean(pairs, axis=1) - target) < 4 * se + 2e-4 * target)

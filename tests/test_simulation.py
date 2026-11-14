"""Monte-Carlo engine against the Appendix D oracle, and model-free identities
(martingale benchmark, exact decomposition) away from the Gaussian case."""
import numpy as np
import jax.numpy as jnp
import pytest

from slidingcurve import Model, decompose, gaussian as g, simulate, time_grid, volscale

MATS = jnp.array([1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
BP = 1e-4


def run(model, curve, key, n_paths=2**15, steps_per_year=12):
    t = time_grid(40.0, steps_per_year)
    ybar = volscale.ybar_zeroth(curve, model.delta, t)
    prof = simulate(model, curve, t, ybar, key, n_paths=n_paths)
    return prof, decompose(prof, model, curve, MATS)


@pytest.fixture(scope="module")
def gauss(ns_curve, base_model, key):
    return run(base_model, ns_curve, key)


def test_gaussian_mean_x_eq87(gauss, ns_curve, base_model):
    prof, _ = gauss
    m = base_model
    exact = g.mean_x_P(ns_curve, m.kappa, m.delta, m.sigma, prof.t)
    mean = jnp.mean(prof.x, axis=0)
    se = jnp.std(prof.x, axis=0, ddof=1) / np.sqrt(prof.x.shape[0])
    # the drift step is exact for the mean in the Gaussian case
    assert np.all(np.abs(mean - exact) < 5 * se + 0.01 * BP)


def test_gaussian_decomposition_matches_closed_form(gauss, ns_curve, base_model):
    _, dec = gauss
    m = base_model
    for j, T in enumerate(np.asarray(dec.T)):
        gam = g.convexity(m.kappa, m.sigma, T)
        yeh = g.yield_eh(ns_curve, m.kappa, m.delta, m.sigma, T)
        th = g.term_premium(ns_curve, m.kappa, m.delta, m.sigma, T)
        # gamma is deterministic here: only quadrature error
        assert abs(dec.gamma[j] - gam) < 0.01 * BP
        assert abs(dec.y_eh[j] - yeh) < 5 * dec.y_eh_se[j] + 0.05 * BP
        assert abs(dec.theta[j] - th) < 5 * dec.theta_se[j] + 0.05 * BP


@pytest.mark.parametrize("b,eta", [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)])
def test_benchmark_yield_is_P_martingale(ns_curve, base_model, key, b, eta):
    """The sliding-curve condition (26) itself: E^P[Y_1(t)] = Y_1(0)."""
    prof, _ = run(base_model._replace(b=b, eta=eta), ns_curve, key)
    mean = jnp.mean(prof.y1, axis=0)
    se = jnp.std(prof.y1, axis=0, ddof=1) / np.sqrt(prof.y1.shape[0])
    assert np.all(np.abs(mean - prof.y1[0, 0]) < 5 * se + 0.01 * BP)


@pytest.mark.parametrize("b,eta", [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)])
def test_decomposition_closes(ns_curve, base_model, key, b, eta):
    """Eq. (12) holds for any volatility: three independent estimators sum
    to the market yield."""
    _, dec = run(base_model._replace(b=b, eta=eta), ns_curve, key)
    assert np.all(np.abs(dec.residual) < 5 * dec.residual_se + 0.05 * BP)


def test_variance_mean_is_one(ns_curve, base_model, key):
    """Zero market price of volatility risk, eq. (63)."""
    prof, _ = run(base_model._replace(eta=0.5), ns_curve, key)
    np.testing.assert_allclose(jnp.mean(prof.z, axis=0), 1.0, atol=0.03)

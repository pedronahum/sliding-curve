"""B4: the Gaussian-twin control variate for yield options."""
import numpy as np
import jax
import pytest

from slidingcurve import Model, gaussian as g, markets, pricing as pr, volscale

CURVE = markets.paper_curve("market")
OFF = np.array([-0.02, -0.005, 0.0, 0.005, 0.02])


def _p(m, cv, ref="mean", n=2**14, seed=0):
    fn = volscale.bootstrap_fn(CURVE, m, volscale.bootstrap_grid(20.0))
    return pr.yield_option_prices(m, CURVE, 20.0, OFF, fn, jax.random.PRNGKey(seed), n, 24, ref, cv)


@pytest.mark.parametrize("ref", ["mean", "forward"])
def test_exact_in_the_gaussian_case(ref):
    """b = 0, eta = 0: the twin is the model, so the estimator has no variance
    and every seed gives the discrete scheme's exact price."""
    m = Model(0.03, 5.0, 0.012)
    a, b = _p(m, True, ref, seed=1), _p(m, True, ref, seed=2)
    np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), atol=1e-9)
    assert np.all(np.asarray(a[2]) < 1e-8)
    # and it is the Bachelier price at the closed-form variance (to the 24-steps-a-year discretisation)
    s = g.A(0.03, 5.0) * 0.012 * np.sqrt(float(g.B(0.06, 20.0)))
    k = np.asarray(pr.yield_option_moneyness(CURVE, m, 20.0, OFF, a[0], ref))
    from scipy.stats import norm
    exact = -k * norm.cdf(-k / s) + s * norm.pdf(k / s)
    np.testing.assert_allclose(np.asarray(a[1]), exact, rtol=2e-4)


@pytest.mark.parametrize("b,eta", [(0.367, 0.0), (1.0, 0.0), (0.367, 0.5)])
def test_unbiased_against_the_plain_estimator(b, eta):
    m = Model(0.03, 5.0, 0.012, b=b, eta=eta)
    _, p0, s0 = _p(m, False, n=2**15, seed=3)
    _, p1, s1 = _p(m, True, n=2**15, seed=3)
    assert np.all(np.abs(np.asarray(p1 - p0)) < 4 * np.asarray(s0))


def test_reduces_the_error_at_the_market_skew():
    m = Model(0.03, 5.0, 0.012, b=0.367)
    assert np.all(np.asarray(_p(m, False)[2]) > 3 * np.asarray(_p(m, True)[2]))

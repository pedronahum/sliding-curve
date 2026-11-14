"""B1: the lognormal variance scheme, smooth in eta, against QE and exact
CIR moments, and autodiff in eta through the pricer."""
import numpy as np
import jax
import jax.numpy as jnp

from slidingcurve import Model, bachelier as bl, calibration, markets, pricing as pr, volscale
from slidingcurve.model import variance_step


def test_lognormal_step_matches_conditional_moments():
    m = Model(0.03, 5.0, 0.01, eta=1.5, kappa_z=0.1, variance_scheme="lognormal")
    dt = 1.0 / 24
    xi = jax.random.normal(jax.random.PRNGKey(0), (2**20,))
    for z0 in (0.05, 1.0, 4.0):
        z1 = np.asarray(variance_step(m, jnp.full(xi.shape, z0), xi, dt))
        e = np.exp(-0.1 * dt)
        mean = 1.0 + (z0 - 1.0) * e
        var = 1.5**2 * (z0 * e * (1 - e) / 0.1 + 0.5 * 0.1 * ((1 - e) / 0.1) ** 2)    # CIR, theta = 1
        np.testing.assert_allclose(z1.mean(), mean, rtol=4 * np.sqrt(var) / mean / 2**10)
        np.testing.assert_allclose(z1.var(), var, rtol=0.05)
        assert z1.min() > 0


def _price(m, eta, n=2**13):
    curve = markets.paper_curve("market")
    fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
    return pr.yield_option_prices(m._replace(eta=eta), curve, 20.0, jnp.array([-0.02, 0.0, 0.02]), fn,
                                  jax.random.PRNGKey(5), n)[1]


def test_autodiff_in_eta_equals_finite_differences_for_the_lognormal_scheme():
    m = Model(0.03, 5.0, 0.01492, b=0.417, eta=0.779, variance_scheme="lognormal")
    ad = np.asarray(jax.jacfwd(lambda e: _price(m, e))(0.779))
    h = 1e-4
    fd = (np.asarray(_price(m, 0.779 + h)) - np.asarray(_price(m, 0.779 - h))) / (2 * h)
    np.testing.assert_allclose(ad, fd, rtol=2e-3)


def test_lognormal_and_qe_smiles_agree():
    """At the paper's three-parameter point (b = 0.417, eta = 0.779)."""
    m = Model(0.03, 5.0, 0.01492, b=0.417, eta=0.779)
    off = np.array([-0.02, 0.0, 0.02])
    v = {}
    for sch in ("qe", "lognormal"):
        p = np.asarray(_price(m._replace(variance_scheme=sch), 0.779, 2**15))
        v[sch] = 1e4 * bl.implied_normal_vol(p, 0.0, off, 20.0)
    np.testing.assert_allclose(v["lognormal"], v["qe"], atol=0.5)      # s.e. ~0.8bp each, common rate noise


def test_eta_calibration_uses_autodiff_and_recovers_a_known_eta():
    curve = markets.paper_curve("market")
    off = np.array([-0.02, -0.005, 0.0, 0.005, 0.02])
    true = Model(0.03, 5.0, 0.012, b=0.3, eta=0.7, variance_scheme="lognormal")
    key = jax.random.PRNGKey(3)
    fn = volscale.bootstrap_fn(curve, true, np.arange(1.0, 21.0))
    _, p, _ = pr.yield_option_prices(true, curve, 20.0, off, fn, key, 2**13)
    vols = bl.implied_normal_vol(np.asarray(p), 0.0, off, 20.0)
    fit = calibration.calibrate_yield_smile(true._replace(sigma=0.011, b=0.4, eta=0.5), curve, 20.0, off, vols,
                                            free=("sigma", "b", "eta"), key=key, n_paths=2**13)
    np.testing.assert_allclose([fit.model.sigma, fit.model.b, fit.model.eta], [0.012, 0.3, 0.7], rtol=2e-3)

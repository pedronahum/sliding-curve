"""C1: multi-factor quasi-Gaussian model (Sections 5-6, Appendix C)."""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from slidingcurve import gaussian as g, markets, multifactor as mf, time_grid, volscale

CURVE = markets.paper_curve("baseline")
ONE = mf.MultiModel(kappa=[0.03], deltas=[5.0], vols=[float(g.A(0.03, 5.0)) * 0.01], corr=[[1.0]])
TWO = mf.MultiModel(kappa=[0.02, 0.4], deltas=[2.0, 10.0], vols=[0.0090, 0.0075],
                    corr=[[1.0, 0.7], [0.7, 1.0]])


def test_one_factor_closed_forms_reduce_to_appendix_d():
    for T in (5.0, 20.0, 40.0):
        yeh, th, gm = mf.gaussian_decomposition(ONE, CURVE, T)
        np.testing.assert_allclose(float(gm), float(g.convexity(0.03, 0.01, T)), rtol=1e-10)
        np.testing.assert_allclose(float(yeh), float(g.yield_eh(CURVE, 0.03, 5.0, 0.01, T)), atol=1e-10)
        np.testing.assert_allclose(float(th), float(g.term_premium(CURVE, 0.03, 5.0, 0.01, T)), atol=1e-10)
    # (90)
    np.testing.assert_allclose(float(mf.gaussian_mean_x_fwd(ONE, 7.0, 25.0)[0]),
                               float(g.mean_x_fwd(0.03, 0.01, 7.0, 25.0)), rtol=1e-12)


def test_forward_measure_mean_solves_its_ode():
    """d/dt E^{Q^tau}[x] = y 1 - kappa E - V G(t, tau): the multi-factor (89)."""
    k = jnp.asarray(TWO.kappa)
    V = mf.gaussian_V(TWO)
    tau = 30.0
    f = lambda t: mf.gaussian_mean_x_fwd(TWO, t, tau)
    for t in (1.0, 12.0):
        lhs = jax.jacfwd(f)(t)
        rhs = mf.gaussian_y(TWO, t).sum(1) - k * f(t) - V @ g.B(k, tau - t)
        np.testing.assert_allclose(np.asarray(lhs), np.asarray(rhs), atol=1e-14)
    np.testing.assert_allclose(np.asarray(f(tau)), 0.0, atol=1e-16)          # Remark 15


@pytest.mark.parametrize("model", [ONE, TWO], ids=["d=1", "d=2"])
def test_gaussian_monte_carlo_matches_closed_forms(model):
    t = time_grid(40.0)
    p = mf.simulate_multi(model, CURVE, t, mf.ybar_zeroth(model, CURVE, t), jax.random.PRNGKey(0), n_paths=2**12)
    T = jnp.array([10.0, 25.0, 40.0])
    d = mf.decompose_multi(p, model, CURVE, T)
    exact = np.array([[float(v) for v in mf.gaussian_decomposition(model, CURVE, float(x))] for x in T])
    np.testing.assert_allclose(np.asarray(d.y_eh), exact[:, 0], atol=2e-6)
    np.testing.assert_allclose(np.asarray(d.theta), exact[:, 1], atol=2e-6)
    np.testing.assert_allclose(np.asarray(d.gamma), exact[:, 2], rtol=2e-3)
    ex = mf.gaussian_mean_x_P(model, CURVE, t)
    np.testing.assert_allclose(np.asarray(p.x.mean(0)), np.asarray(ex), atol=2e-6)


def test_skew_and_sv_two_factor_martingales_and_closure():
    m = TWO._replace(skews=[0.6, 0.3], eta=0.8)
    t = time_grid(30.0)
    ybar = mf.piecewise(np.arange(1.0, 31.0), mf.ybar_bootstrap_multi(m, CURVE, np.arange(1.0, 31.0)), t)
    p = mf.simulate_multi(m, CURVE, t, ybar, jax.random.PRNGKey(1), n_paths=2**13)
    Y0 = np.asarray(mf.fwd_benchmarks(CURVE, jnp.array([0.0, 2.0, 10.0]), jnp.zeros(1))[0])
    se = np.asarray(p.Y.std(0, ddof=1)) / 4
    assert np.all(np.abs(np.asarray(p.Y.mean(0)) - Y0[None]) < 4 * se + 1e-9)        # (26)
    d = mf.decompose_multi(p, m, CURVE, jnp.array([5.0, 15.0, 30.0]))
    assert np.all(np.abs(np.asarray(d.residual)) < 3 * np.asarray(d.residual_se) + 2e-6)


def test_benchmark_correlation():
    """Increments of the benchmark yields over one step have covariance S X S dt."""
    t = jnp.linspace(0.0, 1.0 / 12, 2)
    m = TWO
    p = mf.simulate_multi(m, CURVE, t, mf.ybar_zeroth(m, CURVE, t), jax.random.PRNGKey(2), n_paths=2**16)
    # V = M^-1 S X S M^-T, so M V M' = S X S
    M, _, _, _, _ = mf.structure(m)
    SXS = M @ p.V.mean(0)[0] @ M.T
    s = jnp.asarray(m.vols)
    np.testing.assert_allclose(np.asarray(SXS), np.asarray(s[:, None] * jnp.asarray(m.corr) * s[None, :]), rtol=1e-12)


def test_general_bootstrap():
    grid = np.arange(1.0, 21.0)
    for weighted in (False, True):
        one = mf.ybar_bootstrap_multi(ONE._replace(skews=[0.5]), CURVE, grid, weighted=weighted)[:, 0]
        ref = volscale.ybar_bootstrap(CURVE, 0.03, 5.0, 0.01, grid, weighted=weighted)
        np.testing.assert_allclose(np.asarray(one), np.asarray(ref), atol=1e-13)
    # with (78) in place of the closed form, the solution is (79): the forwards
    m = TWO._replace(skews=[0.5, 0.2])
    zb = mf.ybar_bootstrap_multi(m, CURVE, grid, weighted=True, zeroth_mean=True)
    np.testing.assert_allclose(np.asarray(zb), np.asarray(mf.ybar_zeroth(m, CURVE, jnp.asarray(grid))), atol=1e-13)

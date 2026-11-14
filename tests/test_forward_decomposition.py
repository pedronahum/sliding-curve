"""C3: forward-rate decomposition (20) and Appendix F."""
import numpy as np
import jax
import jax.numpy as jnp

from slidingcurve import (Curve, Model, decompose, forward as fw, gaussian as g, holee, markets,
                          simulate, time_grid, volscale)

KAPPA, SIG = 0.03, 0.01


def test_futures_convexity_closed_form_equals_risk_neutral_drift():
    """(105) against E^Q[x(t)] from (41) by quadrature: eq. (24)."""
    for t in (1.0, 10.0, 30.0):
        np.testing.assert_allclose(float(fw.mean_x_Q(KAPPA, SIG, t)), float(fw.futures_convexity(KAPPA, SIG, t)),
                                   rtol=1e-6)
    # Proposition 5 / F.3: (1/T) int gamma_f = gamma(T), eq. (94)
    for T in (5.0, 40.0):
        s = np.linspace(0, T, 4001)
        avg = np.trapezoid(np.asarray(fw.futures_convexity(KAPPA, SIG, jnp.asarray(s))), s) / T
        np.testing.assert_allclose(avg, float(g.convexity(KAPPA, SIG, T)), rtol=1e-6)


# Maturities stay below 40y: pi(T) weights the drift at t = T fully, and at
# t = 40 that drift needs f(0, 45), the last node of the paper curve, where
# the forward jumps to the flat extrapolation (an O(dt) end effect of the
# curve, not of the decomposition).


def _run(b=0.0, eta=0.0):
    curve = markets.paper_curve("baseline")
    m = Model(kappa=KAPPA, delta=5.0, sigma=SIG, b=b, eta=eta)
    t = time_grid(40.0)
    ybar = volscale.bootstrap_fn(curve, m, np.arange(1.0, 41.0))(t)
    prof = simulate(m, curve, t, ybar, jax.random.PRNGKey(2), n_paths=2**14)
    return curve, m, prof


def test_forward_decomposition_gaussian_closed_forms():
    curve, m, prof = _run()
    T = jnp.array([5.0, 20.0, 39.0])
    d = fw.decompose_forward(prof, m, curve, T)
    np.testing.assert_allclose(np.asarray(d.gamma_f), np.asarray(fw.futures_convexity(KAPPA, SIG, T)), rtol=2e-3)
    exp_r = np.asarray(curve.forward(T)) + np.asarray(g.mean_x_P(curve, KAPPA, 5.0, SIG, T))
    np.testing.assert_allclose(np.asarray(d.exp_rate), exp_r, atol=5e-6)
    assert np.all(np.abs(np.asarray(d.residual)) < 1e-5)


def test_forward_decomposition_closes_and_averages_to_yield_decomposition_with_skew_and_sv():
    """(20) closes, and Proposition 5: gamma = avg gamma_f, theta = avg pi."""
    curve, m, prof = _run(b=0.75, eta=0.8)
    T = jnp.arange(1.0, 40.0)
    d = fw.decompose_forward(prof, m, curve, T)
    assert np.all(np.abs(np.asarray(d.residual)) < 3 * np.asarray(d.residual_se) + 2e-5)
    y = decompose(prof, m, curve, jnp.array([39.0]))
    dense = fw.decompose_forward(prof, m, curve, prof.t[:469])
    tt = np.asarray(prof.t[:469])
    np.testing.assert_allclose(np.trapezoid(np.asarray(dense.gamma_f), tt) / 39.0, float(y.gamma[0]), rtol=2e-3)
    np.testing.assert_allclose(np.trapezoid(np.asarray(dense.pi), tt) / 39.0, float(y.theta[0]), rtol=2e-3)


def test_proposition_17_constant_mpr_curve():
    """Curve (106) has constant sliding-curve MPR xi; with kappa = 0 the MPR
    of a general curve is f'(0,t)/sigma + sigma t (Appendix E)."""
    r0, xi = 0.03, 0.1
    for kappa in (0.0, 0.03):
        c = Curve(lambda T: jnp.where(T > 1e-8, _int_fwd(r0, xi, kappa, SIG, T) / jnp.maximum(T, 1e-8), r0))
        lam = np.asarray(fw.sliding_curve_mpr(c, kappa, SIG, jnp.array([1.0, 10.0, 30.0])))
        np.testing.assert_allclose(lam, xi, atol=1e-8)
    ns = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
    t = jnp.array([2.0, 15.0])
    np.testing.assert_allclose(np.asarray(fw.sliding_curve_mpr(ns, 0.0, SIG, t)),
                               np.asarray(holee.mpr_martingale(ns, SIG, t)), rtol=1e-12)


def _int_fwd(r0, xi, kappa, sigma, T):
    """int_0^T of (106), in closed form."""
    if kappa == 0.0:
        return r0 * T + xi * sigma * T**2 / 2 - sigma**2 * T**3 / 6
    e = jnp.exp(-kappa * T)
    intB = (T - (1 - e) / kappa) / kappa
    intB2 = (T - 2 * (1 - e) / kappa + (1 - e * e) / (2 * kappa)) / kappa**2
    return r0 * T + xi * sigma * intB - 0.5 * sigma**2 * intB2


def test_morton_regression_recovers_the_constant_mpr_curve():
    """Remark 18: with kappa = 0 the curve (106) is exactly (102), so Morton's
    regression of Fut = Fwd + sigma^2 T^2 / 2 on sigma T returns r(0) and xi,
    and (101) gives the Ho-Lee constant-MPR term premium xi sigma T / 2."""
    r0, xi = 0.03, 0.09
    T = jnp.arange(7.0, 31.0)
    fwd = fw.constant_mpr_forward(r0, xi, 0.0, SIG, T)
    fut = fw.morton_futures(fwd, SIG, T)
    r_inf, xi_hat = fw.morton_regression(T, fut, jnp.full_like(T, SIG))
    np.testing.assert_allclose([float(r_inf), float(xi_hat)], [r0, xi], atol=1e-12)
    np.testing.assert_allclose(float(fw.morton_term_premium(xi, lambda t: SIG + 0 * t, 20.0)),
                               holee.term_premium_const(SIG, xi, 20.0), rtol=1e-9)

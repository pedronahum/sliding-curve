"""C2: Ho-Lee closed forms of Appendix E, against the paper's printed
formulas, the general Gaussian closed forms of Appendix D (kappa = 0,
delta -> 0) and Monte Carlo."""
import numpy as np
import jax
import jax.numpy as jnp

from slidingcurve import Curve, Model, decompose, gaussian as g, holee, simulate, time_grid, volscale

CURVE = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
SIG = 0.01


def test_printed_formulas():
    T = np.array([5.0, 10.0, 30.0])
    np.testing.assert_allclose(holee.convexity(SIG, T), SIG**2 * T**2 / 6)
    np.testing.assert_allclose(holee.term_premium_const(SIG, 0.2, T), 0.5 * 0.2 * SIG * T)
    # E.1: Y(0,T) = Y_eh + lambda sigma T / 2 - sigma^2 T^2 / 6
    y = CURVE.zero(T)
    np.testing.assert_allclose(holee.yield_eh_const(CURVE, SIG, 0.2, T) + holee.term_premium_const(SIG, 0.2, T)
                               - holee.convexity(SIG, T), y, atol=1e-15)
    # E.2 double check: theta = (sigma / T) int_0^T lambda(t) (T - t) dt
    for TT in T:
        t = np.linspace(0, TT, 20001)
        lam = np.asarray(holee.mpr_martingale(CURVE, SIG, jnp.asarray(t)))
        direct = SIG / TT * np.trapezoid(lam * (TT - t), t)
        np.testing.assert_allclose(direct, float(holee.term_premium_martingale(CURVE, SIG, TT)), rtol=1e-6)
    # E.3: R_Pi1 - R_Pi2 = theta - 2 gamma, and the paper's 30y example:
    # sigma = 100bp, T = 30 -> 2 gamma = 300bp
    np.testing.assert_allclose(2 * holee.convexity(0.01, 30.0), 0.03)
    np.testing.assert_allclose(holee.growth_rate_gap(CURVE, SIG, 30.0),
                               holee.term_premium_martingale(CURVE, SIG, 30.0) - 2 * holee.convexity(SIG, 30.0))


def test_holee_is_the_kappa_zero_short_benchmark_limit_of_appendix_d():
    """D.5: with delta -> 0 the benchmark is the short rate and Appendix D
    reduces to E.2."""
    for T in (5.0, 20.0):
        np.testing.assert_allclose(float(g.convexity(0.0, SIG, T)), holee.convexity(SIG, T), rtol=1e-10)
        np.testing.assert_allclose(float(g.yield_eh(CURVE, 0.0, 1e-4, SIG, T)),
                                   float(holee.yield_eh_martingale(CURVE, T)), atol=1e-6)   # O(delta)
        np.testing.assert_allclose(float(g.term_premium(CURVE, 0.0, 1e-4, SIG, T)),
                                   float(holee.term_premium_martingale(CURVE, SIG, T)), atol=1e-6)


def test_monte_carlo_short_rate_martingale():
    """The library simulator at kappa = 0 with a tiny benchmark tenor is the
    E.2 model: Y_eh = r(0), theta = roll + sigma^2 T^2 / 6."""
    m = Model(kappa=0.0, delta=1e-3, sigma=SIG)
    t = time_grid(30.0, steps_per_year=24)
    d = decompose(simulate(m, CURVE, t, volscale.ybar_zeroth(CURVE, m.delta, t), jax.random.PRNGKey(3),
                           n_paths=2**14), m, CURVE, jnp.array([10.0, 30.0]))
    np.testing.assert_allclose(np.asarray(d.y_eh), float(CURVE.forward(0.0)), atol=3e-6 + 3 * float(d.y_eh_se.max()))
    np.testing.assert_allclose(np.asarray(d.gamma), holee.convexity(SIG, np.array([10.0, 30.0])), rtol=1e-3)


def test_monte_carlo_constant_mpr_and_cash_growth():
    """E.1 and E.3 by direct simulation of r: E^P int r = T Y_eh(T) under a
    constant MPR; (1/T) log E^P exp(X) = r0 + sigma^2 T^2 / 6 under E.2."""
    rng = np.random.default_rng(0)
    T, n, lam = 20.0, 1000, 0.25
    t = np.linspace(0, T, n + 1)
    dW = rng.standard_normal((200_000, n)) * np.sqrt(T / n)
    W = np.concatenate([np.zeros((200_000, 1)), np.cumsum(dW, 1)], 1)
    f0 = np.asarray(CURVE.forward(jnp.asarray(t)))
    r = f0 + 0.5 * SIG**2 * t**2 - SIG * lam * t + SIG * W
    X = np.trapezoid(r, t, axis=1)
    np.testing.assert_allclose(X.mean() / T, float(holee.yield_eh_const(CURVE, SIG, lam, T)),
                               atol=4 * X.std() / T / np.sqrt(len(X)))
    r2 = float(CURVE.forward(0.0)) + SIG * W
    X2 = np.trapezoid(r2, t, axis=1)
    np.testing.assert_allclose(np.log(np.mean(np.exp(X2))) / T, float(holee.growth_rate_cash(CURVE, SIG, T)),
                               atol=2e-5)
    np.testing.assert_allclose(X2.var(), SIG**2 * T**3 / 3, rtol=0.02)     # (72): Var X = 2 T gamma

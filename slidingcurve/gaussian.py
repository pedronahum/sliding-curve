"""Linear Gaussian one-factor closed forms (Appendix D): b = 0, eta = 0.

These are the oracle for the simulation engine. Integrals over the initial
curve are done by Gauss-Legendre quadrature so the formulas hold for any
curve and any kappa >= 0; the paper's flat-curve expressions (93), (94), (96)
and the kappa = 0 case (D.5) are checked against them in the tests.
"""
from collections.abc import Callable

import numpy as np
import jax.numpy as jnp

from ._typing import Array, ArrayLike, FloatSeq
from .curve import Curve, _phi

_GL_X, _GL_W = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(96))

# Two typing notes. (i) numpy's stubs widen `ndarray * jax.Array` to an ndarray
# and jax's type jnp.sum as Any, so the closed forms below wrap results that
# are already jax arrays in jnp.asarray (the identity on an array, and jax wins
# every mixed operation at runtime). (ii) the quadrature interval and the times
# fed to it are `float | Array`, not ArrayLike: mypy checks every combination of
# two union operands and ArrayLike holds both bool and np.bool_, whose __sub__
# overloads do not meet, so `b - a` and `tau - t` would be rejected.


def _quad(fn: Callable[[Array], ArrayLike], a: float | Array, b: float | Array) -> Array:
    """Integral of fn over [a, b] (scalars), Gauss-Legendre."""
    h = 0.5 * (b - a)
    return jnp.asarray(h * jnp.sum(_GL_W * fn(a + h * (_GL_X + 1.0))))


def B(kappa: ArrayLike, s: ArrayLike) -> Array:
    """B(s) = (1 - exp(-kappa s)) / kappa, eq. (83); equals s at kappa = 0."""
    return jnp.asarray(s * _phi(kappa * s))


def A(kappa: ArrayLike, delta: ArrayLike) -> Array:
    """A(delta) = B(delta) / delta, eq. (83)."""
    return _phi(kappa * delta)


def y_det(kappa: ArrayLike, sigma: ArrayLike, t: ArrayLike) -> Array:
    """Deterministic auxiliary variance y(t), eq. (85)."""
    return jnp.asarray(sigma**2 * B(2.0 * kappa, t))


def mean_x_P(curve: Curve, kappa: ArrayLike, delta: ArrayLike, sigma: ArrayLike,
             t: ArrayLike) -> Array:
    """E^P[x(t)] under the sliding-curve measure, eq. (87)."""
    return ((curve.fwd_yield(0.0, delta) - curve.fwd_yield(t, t + delta)) / A(kappa, delta)
            - 0.5 * B(kappa, delta) * y_det(kappa, sigma, t))


def mean_x_fwd(kappa: ArrayLike, sigma: ArrayLike, t: float | Array, tau: ArrayLike) -> Array:
    """I(t, tau) = E^{Q^tau}[x(t)] = -y(t) B(tau - t), eq. (90)."""
    return -y_det(kappa, sigma, t) * B(kappa, tau - t)


def yhat(curve: Curve, kappa: ArrayLike, delta: ArrayLike, sigma: ArrayLike, t: float | Array,
         Tm: ArrayLike) -> Array:
    """Yhat_1(t, T_m): forward-measure mean of the benchmark yield, eq. (91)."""
    Bd = B(kappa, delta)
    return (curve.fwd_yield(t, t + delta)
            + A(kappa, delta) * mean_x_fwd(kappa, sigma, t, Tm + delta)
            + Bd**2 / (2.0 * delta) * y_det(kappa, sigma, t))


def convexity(kappa: ArrayLike, sigma: ArrayLike, T: float | Array) -> Array:
    """gamma(T) = sigma^2 / (2T) * int_0^T B(T - t)^2 dt, eq. (94)."""
    return jnp.asarray(sigma**2 / (2.0 * T) * _quad(lambda t: B(kappa, T - t) ** 2, 0.0, T))


def yield_eh(curve: Curve, kappa: ArrayLike, delta: ArrayLike, sigma: ArrayLike,
             T: float | Array) -> Array:
    """Expectation-hypothesis yield Y_eh(T), eq. (92)."""
    return curve.zero(T) + _quad(
        lambda t: mean_x_P(curve, kappa, delta, sigma, t), 0.0, T) / T


def term_premium(curve: Curve, kappa: ArrayLike, delta: ArrayLike, sigma: ArrayLike,
                 T: float | Array) -> Array:
    """theta(T) = Y(0, T) + gamma(T) - Y_eh(T), eq. (95)."""
    return (curve.zero(T) + convexity(kappa, sigma, T)
            - yield_eh(curve, kappa, delta, sigma, T))


def payer_swaption(curve: Curve, kappa: float, sigma: float, expiry: float,
                   pay_times: ArrayLike | FloatSeq, accruals: ArrayLike | FloatSeq,
                   strike: float) -> float:
    """Exact payer swaption price in the Gaussian model (Jamshidian's
    decomposition into zero-coupon bond puts). Host-side; the oracle for the
    Monte-Carlo pricer."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    Tn = np.asarray(pay_times, dtype=float)
    c = strike * np.asarray(accruals, dtype=float)
    c[-1] += 1.0
    g = np.asarray(B(kappa, Tn - expiry))
    y = float(y_det(kappa, sigma, expiry))
    p0T = float(curve.discount(expiry))
    p0 = np.asarray(curve.discount(jnp.asarray(Tn)))
    bonds = lambda x: p0 / p0T * np.exp(-g * x - 0.5 * g * g * y)
    x_star = brentq(lambda x: np.sum(c * bonds(x)) - 1.0, -1.0, 1.0, xtol=1e-15)
    k = bonds(x_star)
    sp = sigma * g * np.sqrt(float(B(2.0 * kappa, expiry)))    # stdev of log P(T, T_i)
    d1 = np.log(p0 / (k * p0T)) / sp + 0.5 * sp
    return float(np.sum(c * (k * p0T * norm.cdf(-(d1 - sp)) - p0 * norm.cdf(-d1))))

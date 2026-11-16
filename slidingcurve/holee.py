"""Ho-Lee closed forms, Appendix E.

The Ho-Lee model is the linear Gaussian model of Appendix D with kappa = 0:
dx = sigma^2 t dt + sigma dW^Q, r = f(0, t) + x. Two physical measures:

E.1  constant market price of risk lambda, dW^Q = dW^P - lambda dt;
E.2  the martingale condition imposed on the short rate (benchmark tenor
     delta -> 0), lambda(t) = f'(0, t) / sigma + sigma t.

E.3 checks the "alternative viewpoint" of Appendix B (Proposition 9) on the
second case. All quantities are in rate units (not basis points); `curve`
is a slidingcurve.Curve.
"""
import jax
import jax.numpy as jnp

from ._typing import Array, ArrayLike
from .curve import Curve

# The pure-arithmetic closed forms below are declared `-> ArrayLike`: they are
# plain Python/numpy arithmetic, so the result follows the inputs (a float for
# scalars, an ndarray or a jax array for array arguments). Wrapping them in
# jnp.asarray would satisfy a narrower `-> Array` but would change what they
# return, so the honest alias is used instead.


def convexity(sigma: ArrayLike, T: ArrayLike) -> ArrayLike:
    """gamma(T) = sigma^2 T^2 / 6 (both measures; the convexity is a pure
    volatility quantity)."""
    return sigma**2 * T**2 / 6.0


# --- E.1: constant market price of risk ---------------------------------------

def term_premium_const(sigma: ArrayLike, lam: ArrayLike, T: ArrayLike) -> ArrayLike:
    """theta(T) = lambda sigma T / 2."""
    return 0.5 * lam * sigma * T


def excess_return_const(sigma: ArrayLike, lam: ArrayLike, t: ArrayLike,
                        T: ArrayLike) -> ArrayLike:
    """nu^P(t, T) = lambda sigma (T - t)."""
    # ArrayLike - ArrayLike: mypy tries every combination of the union members,
    # and bool.__sub__ rejects np.bool_; both operands are times here.
    return lam * sigma * (T - t)    # type: ignore[operator]


def yield_eh_const(curve: Curve, sigma: ArrayLike, lam: ArrayLike, T: ArrayLike) -> Array:
    """Y_eh(T) = Y(0, T) - lambda sigma T / 2 + sigma^2 T^2 / 6, eq. (98)."""
    return curve.zero(T) - 0.5 * lam * sigma * T + convexity(sigma, T)


def mean_short_rate_const(curve: Curve, sigma: ArrayLike, lam: ArrayLike, t: ArrayLike) -> Array:
    """E^P[r(t)] = f(0, t) + sigma^2 t^2 / 2 - sigma lambda t, eq. (98)."""
    return curve.forward(t) + 0.5 * sigma**2 * t**2 - sigma * lam * t


# --- E.2: martingale condition on the short rate -------------------------------

def mpr_martingale(curve: Curve, sigma: ArrayLike, t: ArrayLike) -> Array:
    """lambda(t) = f'(0, t) / sigma + sigma t."""
    fprime = jnp.vectorize(jax.grad(lambda s: curve.forward(s)))(jnp.asarray(t, dtype=float))
    # wrapped: jnp.vectorize is Any in jax's stubs, and fprime is already an array
    return jnp.asarray(fprime / sigma + sigma * t)


def yield_eh_martingale(curve: Curve, T: ArrayLike) -> Array:
    """Y_eh(T) = r(0)."""
    return curve.forward(0.0) + 0.0 * jnp.asarray(T, dtype=float)


def term_premium_martingale(curve: Curve, sigma: ArrayLike, T: ArrayLike) -> Array:
    """theta(T) = (Y(0, T) - r(0)) + sigma^2 T^2 / 6: roll plus convexity."""
    return curve.zero(T) - curve.forward(0.0) + convexity(sigma, T)


# --- E.3: the alternative viewpoint (Appendix B) -------------------------------

def int_rate_moments_martingale(curve: Curve, sigma: ArrayLike,
                                T: ArrayLike) -> tuple[Array, ArrayLike]:
    """X = int_0^T r dt under P with r = r(0) + sigma W^P: E^P X = r(0) T,
    Var^P X = sigma^2 T^3 / 3 = 2 T gamma(T), eq. (72)."""
    return curve.forward(0.0) * T, sigma**2 * T**3 / 3.0


def growth_rate_cash(curve: Curve, sigma: ArrayLike, T: ArrayLike) -> Array:
    """R_Pi2(0, T) = (E^P X + Var^P X / 2) / T = r(0) + sigma^2 T^2 / 6."""
    m, v = int_rate_moments_martingale(curve, sigma, T)
    return (m + 0.5 * v) / T


def growth_rate_gap(curve: Curve, sigma: ArrayLike, T: ArrayLike) -> Array:
    """R_Pi1 - R_Pi2 = Y(0, T) - r(0) - sigma^2 T^2 / 6 = theta - 2 gamma:
    the trap of Remark 10."""
    return curve.zero(T) - curve.forward(0.0) - convexity(sigma, T)

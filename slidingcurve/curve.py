"""Initial yield curve.

A curve is defined by one scalar function, the continuously compounded zero
yield Y(0, T). Everything else follows from it:

    -log P(0, T) = T Y(0, T)
    f(0, T)      = d/dT [T Y(0, T)]                 (autodiff, no formula)
    Y(0, T, M)   = (M Y(0, M) - T Y(0, T)) / (M - T)

so a new curve family is one function, and the forwards are consistent with
the yields by construction.
"""
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp


def _phi(u):
    """(1 - exp(-u)) / u, safe (value and gradient) at u = 0."""
    small = jnp.abs(u) < 1e-6
    us = jnp.where(small, 1.0, u)
    return jnp.where(small, 1.0 - 0.5 * u, -jnp.expm1(-us) / us)


class Curve:
    def __init__(self, zero_fn: Callable):
        """zero_fn: scalar T -> Y(0, T), written in jax.numpy, finite at T = 0."""
        self._zero = zero_fn
        self._neglogdf = lambda T: T * zero_fn(T)
        self._fwd = jax.grad(self._neglogdf)

    # --- evaluation (all broadcast over array arguments) -------------------
    def zero(self, T):
        return jnp.vectorize(self._zero)(jnp.asarray(T, dtype=float))

    def forward(self, T):
        """Instantaneous forward rate f(0, T)."""
        return jnp.vectorize(self._fwd)(jnp.asarray(T, dtype=float))

    def discount(self, T):
        return jnp.exp(-jnp.vectorize(self._neglogdf)(jnp.asarray(T, dtype=float)))

    def fwd_yield(self, T, M):
        """Forward term yield Y(0, T, M), M > T."""
        T = jnp.asarray(T, dtype=float)
        M = jnp.asarray(M, dtype=float)
        n = jnp.vectorize(self._neglogdf)
        return (n(M) - n(T)) / (M - T)

    # --- constructors ------------------------------------------------------
    @classmethod
    def flat(cls, rate: float) -> "Curve":
        return cls(lambda T: rate + 0.0 * T)

    @classmethod
    def nelson_siegel(cls, beta0, beta1, beta2, tau, beta3=0.0, tau2=1.0) -> "Curve":
        """Nelson-Siegel zero yields; beta3/tau2 add the Svensson hump."""
        def zero(T):
            u, u2 = T / tau, T / tau2
            return (beta0 + beta1 * _phi(u) + beta2 * (_phi(u) - jnp.exp(-u))
                    + beta3 * (_phi(u2) - jnp.exp(-u2)))
        return cls(zero)

    @classmethod
    def from_zero_nodes(cls, times, zeros) -> "Curve":
        """Natural cubic spline through zero-yield nodes, flat outside them.

        A placeholder for market curves: it interpolates zero yields that
        someone else has bootstrapped. It does not bootstrap swap quotes.
        """
        from scipy.interpolate import CubicSpline
        x = np.asarray(times, dtype=float)
        cs = CubicSpline(x, np.asarray(zeros, dtype=float), bc_type="natural")
        xs, c = jnp.asarray(x), jnp.asarray(cs.c)        # c: [4, n-1]

        def zero(T):
            t = jnp.clip(T, xs[0], xs[-1])
            i = jnp.clip(jnp.searchsorted(xs, t, side="right") - 1, 0, xs.shape[0] - 2)
            d = t - xs[i]
            return ((c[0, i] * d + c[1, i]) * d + c[2, i]) * d + c[3, i]
        return cls(zero)

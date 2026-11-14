"""Forward-rate decomposition (Section 4.3) and the Eurodollar-futures
connection (Appendix F).

    f(0, T) = E^P[r(T)] + pi(T) - gamma_f(T)                          (20)
    gamma_f(T) = E^P int_0^T sigma_f(u, T) . sigma_P(u, T) du          (19)
    pi(T)      = E^P int_0^T nu_f(u, T) du,  nu_f = sigma_f . Lambda

In the one-factor model with constant kappa, sigma_f(u, T) = sigma_r(u)
exp(-kappa (T - u)), sigma_P(u, T) = sigma_r(u) B(T - u) and, by (51),
sigma_r Lambda = phi^Y Lambda / A(delta) (Profiles.mu_y, which is mu^Y / A(delta) under
the martingale condition and follows the refined drift of Sections 5.3-5.4
otherwise). The three terms are therefore quadratures
of the same simulated mean profiles as the yield decomposition, estimated
independently, so `residual` is a genuine check. Proposition 5 links the two
decompositions: gamma(T) = (1/T) int_0^T gamma_f and pi = d(T theta)/dT.

Appendix F: Morton's (2022) construction (99)-(101), the curve shape (106)
for which the sliding-curve market price of risk is constant, and the
market price of risk (107) implied by a forward curve.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gaussian import A, B
from .model import Model, Profiles


class ForwardDecomposition(NamedTuple):
    T: jax.Array
    fwd: jax.Array          # f(0, T)
    exp_rate: jax.Array     # E^P[r(T)]
    pi: jax.Array           # forward term premium
    gamma_f: jax.Array      # forward convexity
    residual: jax.Array
    exp_rate_se: jax.Array
    pi_se: jax.Array
    gamma_f_se: jax.Array
    residual_se: jax.Array


def decompose_forward(profiles: Profiles, model: Model, curve, maturities) -> ForwardDecomposition:
    """`maturities` are snapped to the simulation grid."""
    t = profiles.t
    dt = t[1] - t[0]
    T = jnp.asarray(maturities, dtype=float)
    n = jnp.round(T / dt).astype(int)
    T = t[n]
    i = jnp.arange(t.shape[0])
    w = dt * jnp.where(i[None, :] < n[:, None], 1.0, 0.0)
    w = w.at[:, 0].mul(0.5) + 0.5 * dt * (i[None, :] == n[:, None])
    s = jnp.maximum(T[:, None] - t[None, :], 0.0)
    ek = jnp.exp(-model.kappa * s)
    fwd = curve.forward(T)
    exp_rate = fwd[None, :] + profiles.x[:, n]
    pi = (profiles.mu_y @ (w * ek).T) / A(model.kappa, model.delta)
    gamma_f = profiles.var @ (w * ek * B(model.kappa, s)).T
    resid = exp_rate + pi - gamma_f - fwd[None, :]
    k = profiles.x.shape[0]
    mean = lambda a: jnp.mean(a, axis=0)
    se = lambda a: jnp.std(a, axis=0, ddof=1) / jnp.sqrt(k) if k > 1 else jnp.zeros_like(T)
    return ForwardDecomposition(T, fwd, mean(exp_rate), mean(pi), mean(gamma_f), mean(resid),
                                se(exp_rate), se(pi), se(gamma_f), se(resid))


# --- Gaussian closed forms ------------------------------------------------------

def futures_convexity(kappa, sigma, t):
    """gamma_f(t) = sigma^2 B(t)^2 / 2, eq. (105): with deterministic
    volatility, the futures-forward spread E^Q[r(t)] - f(0, t), eq. (24)."""
    return 0.5 * sigma**2 * B(kappa, t) ** 2


def mean_x_Q(kappa, sigma, t, n=2001):
    """E^Q[x(t)] = int_0^t exp(-kappa (t - s)) y(s) ds by quadrature: the
    futures-forward spread computed from the risk-neutral dynamics (41)
    directly, independently of (105)."""
    s = jnp.linspace(0.0, t, n)
    f = jnp.exp(-kappa * (t - s)) * sigma**2 * B(2.0 * kappa, s)
    return jnp.trapezoid(f, s)


# --- Appendix F: Morton's construction ------------------------------------------

def morton_futures(fwd, sigma_T, T):
    """Fut(T) = Fwd(T) + sigma_T^2 T^2 / 2, eq. (99)."""
    return fwd + 0.5 * sigma_T**2 * T**2


def morton_regression(T, fut, sigma_T):
    """Regress Fut(T) on sigma_T T, eq. (100): returns (r_inf, xi)."""
    X = jnp.stack([jnp.ones_like(T), sigma_T * T], axis=1)
    coef, *_ = jnp.linalg.lstsq(X, fut)
    return coef[0], coef[1]


def morton_term_premium(xi, sigma_t_fn, T, n=2001):
    """theta^M(T) = (1/T) int_0^T xi sigma_t t dt, eq. (101)."""
    t = jnp.linspace(0.0, T, n)
    return jnp.trapezoid(xi * sigma_t_fn(t) * t, t) / T


def constant_mpr_forward(r0, xi, kappa, sigma, T):
    """f(0, T) = r(0) + xi sigma B(T) - sigma^2 B(T)^2 / 2, eq. (106): the
    curve for which the sliding-curve (short-rate martingale) market price of
    risk is constant and equal to xi (Proposition 17)."""
    b = B(kappa, T)
    return r0 + xi * sigma * b - 0.5 * sigma**2 * b**2


def sliding_curve_mpr(curve, kappa, sigma, T):
    """lambda(T) from eq. (107), with the martingale condition on the short
    rate: sigma lambda = g' + kappa g + sigma^2 B (1 - kappa B / 2),
    g(T) = f(0, T) - r(0)."""
    T = jnp.asarray(T, dtype=float)
    g = curve.forward(T) - curve.forward(0.0)
    gp = jnp.vectorize(jax.grad(lambda s: curve.forward(s)))(T)
    b = B(kappa, T)
    return (gp + kappa * g + sigma**2 * b * (1.0 - 0.5 * kappa * b)) / sigma

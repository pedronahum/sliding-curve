"""Yield decomposition Y(0,T) = Y_eh(T) + theta(T) - gamma(T), eq. (12).

In the one-factor model G(t, T) = B(T - t) is deterministic, so expectations
and time integrals commute and (13), (54), (55) reduce to quadratures of the
simulated mean profiles:

    Y_eh(T)  = Y(0,T) + 1/T  int_0^T E[x(t)] dt
    theta(T) = 1/T  int_0^T B(T-t)   E[mu^Y(t)] / A(delta) dt
    gamma(T) = 1/2T int_0^T B(T-t)^2 E[sigma_r(t)^2] dt

The three are estimated independently, so `residual = Y_eh + theta - gamma - Y`
is a genuine check of the engine, in any parameter regime.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gaussian import A, B
from .model import Model, Profiles


class Decomposition(NamedTuple):
    T: jax.Array
    y: jax.Array            # market yield Y(0, T)
    y_eh: jax.Array
    theta: jax.Array
    gamma: jax.Array
    residual: jax.Array
    y_eh_se: jax.Array      # standard errors across batches
    theta_se: jax.Array
    gamma_se: jax.Array
    residual_se: jax.Array


def decompose(profiles: Profiles, model: Model, curve, maturities) -> Decomposition:
    """`maturities` must lie on the simulation grid (they are snapped to it)."""
    t = profiles.t
    dt = t[1] - t[0]
    T = jnp.asarray(maturities, dtype=float)
    n = jnp.round(T / dt).astype(int)                         # [M] grid indices
    T = t[n]
    i = jnp.arange(t.shape[0])
    # Trapezoid weights on [0, T_n] for every maturity: [M, N+1]
    w = dt * jnp.where(i[None, :] < n[:, None], 1.0, 0.0)
    w = w.at[:, 0].mul(0.5) + 0.5 * dt * (i[None, :] == n[:, None])
    Bk = B(model.kappa, jnp.maximum(T[:, None] - t[None, :], 0.0))

    y = curve.zero(T)
    # [K, M] per-batch estimates
    y_eh = y + (profiles.x @ w.T) / T
    theta = (profiles.mu_y @ (w * Bk).T) / (T * A(model.kappa, model.delta))
    gamma = (profiles.var @ (w * Bk**2).T) / (2.0 * T)
    resid = y_eh + theta - gamma - y

    k = profiles.x.shape[0]
    mean = lambda a: jnp.mean(a, axis=0)
    se = lambda a: jnp.std(a, axis=0, ddof=1) / jnp.sqrt(k) if k > 1 else jnp.zeros_like(T)
    return Decomposition(T=T, y=y, y_eh=mean(y_eh), theta=mean(theta), gamma=mean(gamma),
                         residual=mean(resid), y_eh_se=se(y_eh), theta_se=se(theta),
                         gamma_se=se(gamma), residual_se=se(resid))

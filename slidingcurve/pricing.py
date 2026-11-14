"""European swaption pricing in the quasi-Gaussian model (risk-neutral side).

Simulation is in the T-forward measure of the option expiry T. From (41) and
the Girsanov shift sigma_r G(t, T) dt,

    dx = (y - kappa x - sigma_r^2 B(T - t)) dt + sigma_r dW^T
    dy = (sigma_r^2 - 2 kappa y) dt

with the same local/stochastic volatility sigma_r as the physical-measure
simulator (the variance driver z is independent of W, so its dynamics are
unchanged). Bonds at expiry are explicit in the state, eq. (42), so

    payer swaption = P(0, T) E^T[(1 - P(T, T_n) - K sum_i tau_i P(T, T_i))^+].

A caplet on the simply compounded rate L(T, T + delta) -- the paper's
calibration instrument -- is the one-period case.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gaussian import A, B
from .model import Model, local_vol, variance_step


class Swaption(NamedTuple):
    expiry: float
    pay_times: jax.Array        # fixed-leg payment times T_1 < ... < T_n
    accruals: jax.Array         # tau_i

    @classmethod
    def regular(cls, expiry, tenor, frequency=2):
        n = int(round(tenor * frequency))
        return cls(float(expiry), expiry + jnp.arange(1, n + 1) / frequency,
                   jnp.full(n, 1.0 / frequency))

    @classmethod
    def caplet(cls, expiry, delta):
        return cls(float(expiry), jnp.array([expiry + delta]), jnp.array([float(delta)]))


def annuity(curve, swp: Swaption):
    return jnp.sum(swp.accruals * curve.discount(swp.pay_times))


def forward_swap_rate(curve, swp: Swaption):
    return (curve.discount(swp.expiry) - curve.discount(swp.pay_times[-1])) / annuity(curve, swp)


def bond(curve, kappa, T, M, x, y):
    """P(T, M) as a function of the state at T, eq. (42). Broadcasts M against x."""
    g = B(kappa, M - T)
    return curve.discount(M) / curve.discount(T) * jnp.exp(-g * x - 0.5 * g * g * y)


def simulate_terminal(model: Model, curve, expiry, ybar_fn, key, n_paths=2**15,
                      steps_per_year=24):
    """State (x, y) at `expiry` under the expiry-forward measure.

    ybar_fn(t) is the volatility scaling Ybar_1 on an array of times. Paths
    come in antithetic pairs: path i and path i + n_paths/2.
    """
    kappa, delta = model.kappa, model.delta
    n = max(int(round(expiry * steps_per_year)), 1)
    dt = expiry / n
    t = jnp.arange(n) * dt
    Bd, Ad = B(kappa, delta), A(kappa, delta)
    c_y = Bd**2 / (2.0 * delta)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2.0 * kappa * dt), B(2.0 * kappa, dt)
    half = n_paths // 2
    xs = (jnp.arange(n), curve.fwd_yield(t, t + delta), ybar_fn(t), B(kappa, expiry - t))

    def step(state, inp):
        x, y, z = state
        i, y0, yb, g = inp
        vol = local_vol(model, y0 + Ad * x + c_y * y, yb, z)
        var = vol**2
        xi = jax.random.normal(jax.random.fold_in(key, i), (2, half))
        xi = jnp.concatenate([xi, -xi], axis=1)
        # exact for the linear part with volatility frozen over the step
        x_new = x * e1 + (y - var * g) * b1 + vol * jnp.sqrt(b2) * xi[0]
        y_new = y * e2 + var * b2
        z_new = variance_step(model, z, xi[1], dt)
        return (x_new, y_new, z_new), None

    init = (jnp.zeros(2 * half), jnp.zeros(2 * half), jnp.ones(2 * half))
    (x, y, _), _ = jax.lax.scan(step, init, xs)
    return x, y


def payer_payoffs(model, curve, swp: Swaption, strikes, x, y):
    """Undiscounted payoff at expiry per strike and path: [n_strikes, n_paths]."""
    p = bond(curve, model.kappa, swp.expiry, swp.pay_times[:, None], x[None, :], y[None, :])
    floating = 1.0 - p[-1]
    fixed = jnp.sum(swp.accruals[:, None] * p, axis=0)
    return jnp.maximum(floating[None, :] - jnp.asarray(strikes)[:, None] * fixed[None, :], 0.0)


def swaption_prices(model: Model, curve, swp: Swaption, strikes, ybar_fn, key,
                    n_paths=2**15, steps_per_year=24):
    """Payer swaption prices and standard errors, one per strike."""
    x, y = simulate_terminal(model, curve, swp.expiry, ybar_fn, key, n_paths, steps_per_year)
    pay = payer_payoffs(model, curve, swp, strikes, x, y)
    half = pay.shape[1] // 2
    pairs = 0.5 * (pay[:, :half] + pay[:, half:])             # antithetic pairs
    df = curve.discount(swp.expiry)
    return df * jnp.mean(pairs, axis=1), df * jnp.std(pairs, axis=1, ddof=1) / jnp.sqrt(half)

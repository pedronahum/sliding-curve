"""One-factor quasi-Gaussian model in the sliding-curve physical measure.

State (per path): x (rate factor), y (auxiliary variance), z (stochastic
variance multiplier). With constant mean reversion kappa, h(t) = exp(-kappa t),
G(t, T) = B(T - t) and H(t) H^Y(t)^{-1} = 1 / A(delta), the P-dynamics (52) are

    dx = (y - kappa x - mu^Y / A(delta)) dt + sigma_r dW
    dy = (sigma_r^2 - 2 kappa y) dt
    dz = kappa_z (1 - z) dt + eta sqrt(z) dZ,            <dW, dZ> = 0     (46)

with benchmark yield, roll and drift

    Y_1    = Y(0, t, t+delta) + A(delta) x + B(delta)^2 / (2 delta) y      (86)
    rho    = (f(t, t+delta) - r(t)) / delta                                (27)
    mu^Y   = rho + sigma_r^2 B(delta)^2 / (2 delta)                        (45)

and the volatility convention of Section 9.1 (basis-point vol flat in time):

    sigma_r = sqrt(z) sigma ((1 - b) + b Y_1 / Ybar_1(t)).

Discretisation. y is advanced exactly for volatility frozen over the step.
The drift of x is then taken as the increment that keeps the benchmark yield
(86) an exact discrete-time martingale,

    A(delta) E[dx] = -(dY(0, t, t+delta) + B(delta)^2 / (2 delta) dy),

which is the integral of the continuous drift above (the kappa x terms cancel
against the roll) and removes the O(dt) bias a left-point Euler drift picks up
from a steep front end. Only the diffusion term is Euler.

The simulator returns per-time-step *path means* of x, mu^Y and sigma_r^2.
Because G(t, T) is deterministic, those three profiles are all that the yield
decomposition needs for every maturity at once (see decomposition.py).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .gaussian import A, B


class Model(NamedTuple):
    kappa: float            # mean reversion of the rate factor
    delta: float            # benchmark tenor (years)
    sigma: float            # short-rate basis-point volatility at b = 0
    b: float = 0.0          # skew: 0 = Gaussian, 1 = lognormal in Y_1
    eta: float = 0.0        # volatility of variance
    kappa_z: float = 0.1    # mean reversion of variance
    vol_cap: float = 10.0   # cap on sigma_r, in multiples of sigma (Remark 8)


class Profiles(NamedTuple):
    """Path means on the time grid, one row per independent batch: [K, N+1]."""
    t: jax.Array
    x: jax.Array
    mu_y: jax.Array
    var: jax.Array          # sigma_r^2
    y1: jax.Array           # benchmark yield, a P-martingale by construction
    z: jax.Array


def time_grid(t_max: float, steps_per_year: int = 12):
    n = int(round(t_max * steps_per_year))
    return jnp.linspace(0.0, t_max, n + 1)


def simulate(model: Model, curve, t, ybar, key, n_paths: int = 2**16,
             n_batches: int = 16, antithetic: bool = True) -> Profiles:
    """Simulate on the uniform grid `t`; `ybar` is Ybar_1 on that grid.

    Paths are split into `n_batches` independent batches so that standard
    errors of time-integrated quantities can be taken across batches.
    """
    kappa, delta, sigma = model.kappa, model.delta, model.sigma
    dt = t[1] - t[0]
    sqdt = jnp.sqrt(dt)
    Bd, Ad, ekd = B(kappa, delta), A(kappa, delta), jnp.exp(-kappa * delta)
    c_y = Bd**2 / (2.0 * delta)
    decay_y, gain_y = jnp.exp(-2.0 * kappa * dt), B(2.0 * kappa, dt)

    per_batch = n_paths // n_batches
    draw_shape = (2, n_batches, per_batch // 2 if antithetic else per_batch)
    y0 = curve.fwd_yield(t, t + delta)
    t_next = t + dt                                           # one past the end: unused step
    dy0 = curve.fwd_yield(t_next, t_next + delta) - y0
    xs = (jnp.arange(t.shape[0]), curve.forward(t), curve.forward(t + delta),
          y0, dy0, jnp.asarray(ybar))

    def step(state, inp):
        x, y, z = state
        i, f0, f0d, y0, dy0, yb = inp
        zp = jnp.maximum(z, 0.0)                              # full truncation
        y1 = y0 + Ad * x + c_y * y
        # Floor at zero: the affine local vol is absorbed where it vanishes.
        vol = jnp.clip(jnp.sqrt(zp) * sigma * ((1.0 - model.b) + model.b * y1 / yb),
                       0.0, model.vol_cap * sigma)
        var = vol**2
        rho = (f0d + ekd * (x + y * Bd) - f0 - x) / delta
        mu_y = rho + var * c_y
        stats = jnp.stack([jnp.mean(a, axis=1) for a in (x, mu_y, var, y1, z)])

        xi = jax.random.normal(jax.random.fold_in(key, i), draw_shape)
        if antithetic:
            xi = jnp.concatenate([xi, -xi], axis=2)
        y_new = y * decay_y + var * gain_y                    # exact for frozen var
        x_new = x - (dy0 + c_y * (y_new - y)) / Ad + vol * sqdt * xi[0]
        z_new = z + model.kappa_z * (1.0 - zp) * dt + model.eta * jnp.sqrt(zp) * sqdt * xi[1]
        return (x_new, y_new, z_new), stats

    shape = (n_batches, draw_shape[2] * (2 if antithetic else 1))
    init = (jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape))
    _, stats = jax.lax.scan(step, init, xs)                   # [N+1, 5, K]
    s = jnp.transpose(stats, (1, 2, 0))                       # [5, K, N+1]
    return Profiles(t=t, x=s[0], mu_y=s[1], var=s[2], y1=s[3], z=s[4])

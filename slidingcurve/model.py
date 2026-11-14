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

Refinements of the P-drift (Sections 5.3-5.4). Any P-drift of Y_1 pins down
the market price of risk, (37); `simulate` adds to the step above

    front end (35), (37):  E^P[Y_1(t_{i+1})] - E^P[Y_1(t_i)] for a given
                           deterministic path (`ey_path`, e.g. front_end_path),
    equilibrium (5.3):     (exp(-a dt) - 1) (Y_1 - Y*), the exact discrete
                           form of the drift -a (Y_1 - Y*) (`reversion`),

so that E^P[Y_1] follows the path, or Y* + exp(-a t) (Y_1(0) - Y*), exactly
on the grid. Given both, the two drifts add.

The simulator returns per-time-step *path means* of x, phi^Y Lambda (which is
mu^Y under the martingale condition) and sigma_r^2.
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
    variance_scheme: str = "qe"   # "qe" (Andersen) or "lognormal" (smooth in eta)


def local_vol(model: Model, y1, ybar, z):
    """Short-rate volatility sigma_r: affine local volatility in the benchmark
    yield times sqrt(z), floored at zero (the affine local vol is absorbed
    where it vanishes) and capped at vol_cap * sigma (Remark 8). Shared by the
    physical-measure simulator and the pricing simulator."""
    zp = jnp.maximum(z, 1e-12)          # keeps d sqrt / dz finite for autodiff
    lv = model.sigma * ((1.0 - model.b) + model.b * y1 / ybar)
    return jnp.clip(jnp.sqrt(zp) * lv, 0.0, model.vol_cap * model.sigma)


def variance_step(model: Model, z, xi, dt):
    """One step of the variance process (46). `xi` are standard normals.

    Both schemes match the conditional mean and variance of z exactly, so
    E[z] = 1 holds without time-step bias even when eta is large:

    "qe"         Andersen's quadratic-exponential scheme (default). The best
                 match to the true distribution near zero, but it switches
                 branch at psi = 1.5, so at fixed random numbers the sample
                 jumps as eta moves: not pathwise differentiable in eta.
    "lognormal"  moment-matched lognormal (Andersen and Brotherton-Ratcliffe),
                 z' = m exp(-s^2/2 + s xi), s^2 = log(1 + psi). Smooth in eta
                 and z, so calibrations with eta free can use autodiff.
    """
    e = jnp.exp(-model.kappa_z * dt)
    b1 = (1.0 - e) / model.kappa_z
    m = 1.0 + (z - 1.0) * e
    s2 = model.eta**2 * (z * e * b1 + 0.5 * model.kappa_z * b1**2)
    psi = jnp.maximum(s2 / m**2, 1e-12)
    if model.variance_scheme == "lognormal":
        s = jnp.sqrt(jnp.log1p(psi))
        return m * jnp.exp(-0.5 * s * s + s * xi)
    # quadratic branch (psi <= 1.5); arguments clamped so the untaken branch
    # stays finite under autodiff
    pq = jnp.minimum(psi, 1.5)
    b2 = 2.0 / pq - 1.0 + jnp.sqrt(2.0 / pq) * jnp.sqrt(2.0 / pq - 1.0)
    quad = m / (1.0 + b2) * (jnp.sqrt(b2) + xi) ** 2
    # exponential branch (psi > 1.5)
    pe = jnp.maximum(psi, 1.5)
    p0 = (pe - 1.0) / (pe + 1.0)
    u = jnp.clip(jax.scipy.stats.norm.cdf(xi), 0.0, 1.0 - 1e-12)
    expo = jnp.where(u > p0, jnp.log((1.0 - p0) / (1.0 - u)) * m / (1.0 - p0), 0.0)
    return jnp.where(psi <= 1.5, quad, expo)


class Profiles(NamedTuple):
    """Path means on the time grid, one row per independent batch: [K, N+1]."""
    t: jax.Array
    x: jax.Array
    mu_y: jax.Array         # phi^Y Lambda = mu^Y - (P-drift of Y_1), (37); mu^Y under (26)
    var: jax.Array          # sigma_r^2
    y1: jax.Array           # benchmark yield, a P-martingale unless the drift is refined
    z: jax.Array


def time_grid(t_max: float, steps_per_year: int = 12):
    n = int(round(t_max * steps_per_year))
    return jnp.linspace(0.0, t_max, n + 1)


def front_end_path(curve, delta, t, Tc, c, form: str = "exp"):
    """Expected path of the benchmark yield Y_1 under the front-end blend (35),

        E^P[Y_1(t)] = (1 - w(t)) Y(0, 0, delta) + w(t) Y(0, t, t + delta),

    with w(t) = exp(-c t / Tc) ("exp") or max(1 - t / Tc, 0)^c ("power").
    One factor, so d = 1 and the benchmark tenor is [t, t + delta]."""
    t = jnp.asarray(t, dtype=float)
    if form == "exp":
        w = jnp.exp(-c * t / Tc)
    elif form == "power":
        w = jnp.maximum(1.0 - t / Tc, 0.0) ** c
    else:
        raise ValueError(f"unknown blend form {form!r}")
    return (1.0 - w) * curve.fwd_yield(0.0, delta) + w * curve.fwd_yield(t, t + delta)


def simulate(model: Model, curve, t, ybar, key, n_paths: int = 2**16,
             n_batches: int = 16, antithetic: bool = True, snapshot_times=None,
             ey_path=None, reversion=None):
    """Simulate on the uniform grid `t`; `ybar` is Ybar_1 on that grid.

    The P-drift makes Y_1 a martingale (26) by default. `ey_path` (values on
    `t`) makes it a martingale around that deterministic path instead, (37);
    `reversion=(a, Y*)` adds reversion to an equilibrium yield (Section 5.3).

    Paths are split into `n_batches` independent batches so that standard
    errors of time-integrated quantities can be taken across batches.

    Returns the mean Profiles. If `snapshot_times` is given (times on the
    grid), also returns the short rate r = f(0, t) + x of every path at those
    times, shape [n_times, n_paths], for distributional results.
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
    # P-drift refinements: the step increment (exact for the mean on the grid)
    # and the drift rate at t_i (second order, for phi^Y Lambda in the profiles,
    # which the decomposition integrates by the trapezoid rule)
    if ey_path is None:
        dey, rate_ey = jnp.zeros_like(t), jnp.zeros_like(t)
    else:
        ey = jnp.asarray(ey_path, dtype=float)
        dey = jnp.concatenate([ey[1:] - ey[:-1], jnp.zeros(1)])   # last step unused
        rate_ey = jnp.gradient(ey, dt)                            # central differences
    if reversion is None:
        a, ystar, rev_k = 0.0, 0.0, 0.0
    else:
        a, ystar = reversion
        rev_k = jnp.expm1(-a * dt)
    xs = (jnp.arange(t.shape[0]), curve.forward(t), curve.forward(t + delta),
          y0, dy0, jnp.asarray(ybar), dey, rate_ey)

    snap_idx = (jnp.round(jnp.asarray(snapshot_times, dtype=float) / dt).astype(int)
                if snapshot_times is not None else jnp.zeros(0, dtype=int))

    def step(state, inp):
        x, y, z, snap = state
        i, f0, f0d, y0, dy0, yb, de, rde = inp
        snap = jnp.where((snap_idx == i)[:, None, None], (f0 + x)[None], snap)
        y1 = y0 + Ad * x + c_y * y
        vol = local_vol(model, y1, yb, z)
        var = vol**2
        rho = (f0d + ekd * (x + y * Bd) - f0 - x) / delta
        mu_y = rho + var * c_y
        drift_y1 = de + rev_k * (y1 - ystar)                 # P-drift of Y_1 over the step
        lam = mu_y - (rde - a * (y1 - ystar))                 # phi^Y Lambda, (37)
        stats = jnp.stack([jnp.mean(a, axis=1) for a in (x, lam, var, y1, z)])

        xi = jax.random.normal(jax.random.fold_in(key, i), draw_shape)
        if antithetic:
            xi = jnp.concatenate([xi, -xi], axis=2)
        y_new = y * decay_y + var * gain_y                    # exact for frozen var
        x_new = x + (drift_y1 - dy0 - c_y * (y_new - y)) / Ad + vol * sqdt * xi[0]
        z_new = variance_step(model, z, xi[1], dt)
        return (x_new, y_new, z_new, snap), stats

    shape = (n_batches, draw_shape[2] * (2 if antithetic else 1))
    init = (jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape),
            jnp.zeros((snap_idx.shape[0],) + shape))
    (_, _, _, snap), stats = jax.lax.scan(step, init, xs)     # stats: [N+1, 5, K]
    s = jnp.transpose(stats, (1, 2, 0))                       # [5, K, N+1]
    prof = Profiles(t=t, x=s[0], mu_y=s[1], var=s[2], y1=s[3], z=s[4])
    if snapshot_times is None:
        return prof
    return prof, snap.reshape(snap_idx.shape[0], -1)

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

Discretisation. y is advanced exactly for volatility frozen over the step
(vol_scheme="euler", the default) or for the trapezoid of sigma_r^2 between
the start and a predicted end of the step (vol_scheme="pc", see pc_vol).
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
from typing import Literal, NamedTuple, Protocol, cast, overload

import jax
import jax.numpy as jnp

from ._typing import Array, ArrayLike, FloatSeq
from .curve import Curve
from .gaussian import A, B


class VarianceModel(Protocol):
    """The part of a model the variance step (46) reads: satisfied by both
    `Model` and `multifactor.MultiModel`."""

    @property
    def eta(self) -> float: ...

    @property
    def kappa_z(self) -> float: ...

    @property
    def variance_scheme(self) -> str: ...


class Model(NamedTuple):
    kappa: float            # mean reversion of the rate factor
    delta: float            # benchmark tenor (years)
    sigma: float            # short-rate basis-point volatility at b = 0
    b: float = 0.0          # skew: 0 = Gaussian, 1 = lognormal in Y_1
    eta: float = 0.0        # volatility of variance
    kappa_z: float = 0.1    # mean reversion of variance
    vol_cap: float = 10.0   # cap on sigma_r, in multiples of sigma (Remark 8)
    variance_scheme: str = "qe"   # "qe" (Andersen) or "lognormal" (smooth in eta)
    vol_scheme: str = "euler"     # "euler" (sigma_r frozen over a step) or "pc" (predictor-corrector)


def local_vol(model: Model, y1: ArrayLike, ybar: ArrayLike, z: ArrayLike) -> Array:
    """Short-rate volatility sigma_r: affine local volatility in the benchmark
    yield times sqrt(z), floored at zero (the affine local vol is absorbed
    where it vanishes) and capped at vol_cap * sigma (Remark 8). Shared by the
    physical-measure simulator and the pricing simulator."""
    zp = jnp.maximum(z, 1e-12)          # keeps d sqrt / dz finite for autodiff
    lv = model.sigma * ((1.0 - model.b) + model.b * y1 / ybar)
    return jnp.clip(jnp.sqrt(zp) * lv, 0.0, model.vol_cap * model.sigma)


def local_vol_dx(model: Model, y1: ArrayLike, ybar: ArrayLike, z: ArrayLike,
                 A_delta: ArrayLike) -> Array:
    """d sigma_r / d x (through Y_1 = ... + A(delta) x), zero where the floor
    or the cap binds. Used by the Ito correction of the predictor-corrector."""
    zp = jnp.maximum(z, 1e-12)
    raw = jnp.sqrt(zp) * model.sigma * ((1.0 - model.b) + model.b * y1 / ybar)
    active = (raw > 0.0) & (raw < model.vol_cap * model.sigma)
    return jnp.where(active, jnp.sqrt(zp) * model.sigma * model.b * A_delta / ybar, 0.0)


def pc_vol(model: Model, vol: ArrayLike, vol_pred: ArrayLike, dvol_dx: ArrayLike,
           dt: ArrayLike) -> tuple[Array, Array, Array]:
    """Predictor-corrector volatility for one step (Kloeden and Platen,
    Section 15.5, with both weights 1/2).

    Returns (variance for the y update and drift terms, diffusion coefficient,
    Ito drift correction). The diffusion is the average of sigma_r at the
    start and at the predicted end of the step; because the end point
    depends on the same normal, the average alone would converge to the
    Stratonovich integral, so -1/2 sigma_r d sigma_r / dx dt is added to keep
    it Ito (and the noise term zero-mean to leading order). The variance is
    the trapezoid of sigma_r^2."""
    # wrapped: arithmetic on ArrayLike widens to a numpy union in jax's stubs
    return (jnp.asarray(0.5 * (vol**2 + vol_pred**2)),
            jnp.asarray(0.5 * (vol + vol_pred)),
            jnp.asarray(-0.5 * vol * dvol_dx * dt))


def variance_step(model: VarianceModel, z: ArrayLike, xi: ArrayLike, dt: ArrayLike) -> Array:
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
        # wrapped: m carries ArrayLike arithmetic on z, which widens the product
        return jnp.asarray(m * jnp.exp(-0.5 * s * s + s * xi))
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
    # wrapped: quad/expo carry ArrayLike arithmetic, which jnp.where widens
    return jnp.asarray(jnp.where(psi <= 1.5, quad, expo))


class Profiles(NamedTuple):
    """Path means on the time grid, one row per independent batch: [K, N+1]."""
    t: jax.Array
    x: jax.Array
    mu_y: jax.Array         # phi^Y Lambda = mu^Y - (P-drift of Y_1), (37); mu^Y under (26)
    var: jax.Array          # sigma_r^2
    y1: jax.Array           # benchmark yield, a P-martingale unless the drift is refined
    z: jax.Array


def time_grid(t_max: float, steps_per_year: int = 12) -> Array:
    n = int(round(t_max * steps_per_year))
    return jnp.linspace(0.0, t_max, n + 1)


def front_end_path(curve: Curve, delta: float, t: ArrayLike, Tc: float, c: float,
                   form: Literal["exp", "power"] = "exp") -> Array:
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


@overload
def simulate(model: Model, curve: Curve, t: Array, ybar: ArrayLike | FloatSeq, key: Array,
             n_paths: int = 2**16, n_batches: int = 16, antithetic: bool = True,
             snapshot_times: None = None, ey_path: ArrayLike | FloatSeq | None = None,
             reversion: tuple[float, float] | None = None,
             normals: ArrayLike | FloatSeq | None = None) -> Profiles: ...


@overload
def simulate(model: Model, curve: Curve, t: Array, ybar: ArrayLike | FloatSeq, key: Array,
             n_paths: int = 2**16, n_batches: int = 16, antithetic: bool = True,
             snapshot_times: ArrayLike | FloatSeq = ..., ey_path: ArrayLike | FloatSeq | None = None,
             reversion: tuple[float, float] | None = None,
             normals: ArrayLike | FloatSeq | None = None) -> tuple[Profiles, Array]: ...


def simulate(model: Model, curve: Curve, t: Array, ybar: ArrayLike | FloatSeq, key: Array,
             n_paths: int = 2**16, n_batches: int = 16, antithetic: bool = True,
             snapshot_times: ArrayLike | FloatSeq | None = None,
             ey_path: ArrayLike | FloatSeq | None = None,
             reversion: tuple[float, float] | None = None,
             normals: ArrayLike | FloatSeq | None = None) -> Profiles | tuple[Profiles, Array]:
    """Simulate on the uniform grid `t`; `ybar` is Ybar_1 on that grid.

    The P-drift makes Y_1 a martingale (26) by default. `ey_path` (values on
    `t`) makes it a martingale around that deterministic path instead, (37);
    `reversion=(a, Y*)` adds reversion to an equilibrium yield (Section 5.3).

    Paths are split into `n_batches` independent batches so that standard
    errors of time-integrated quantities can be taken across batches.

    `normals` replaces the pseudo-random draws, e.g. randomised quasi-random
    ones from qmc.sobol_normals: shape [len(t), 2, n_batches, n_paths //
    n_batches] (step, driver: rate then variance, batch, path). `key` and
    `antithetic` are then ignored.

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
        # cast: jnp.gradient is stubbed Array | list[Array] for a general axis
        rate_ey = cast(Array, jnp.gradient(ey, dt))               # central differences
    a: float
    ystar: float
    rev_k: float | Array
    if reversion is None:
        a, ystar, rev_k = 0.0, 0.0, 0.0
    else:
        a, ystar = reversion
        rev_k = jnp.expm1(-a * dt)
    if normals is not None:
        normals = jnp.asarray(normals, dtype=float)
        antithetic = False
        draw_shape = (2, n_batches, per_batch)
        if normals.shape != (t.shape[0],) + draw_shape:
            raise ValueError(f"normals must have shape {(t.shape[0],) + draw_shape}, got {normals.shape}")
    else:
        normals = jnp.zeros((t.shape[0], 0))                  # unused placeholder
    use_given = normals.ndim == 4
    xs = (jnp.arange(t.shape[0]), curve.forward(t), curve.forward(t + delta),
          y0, dy0, jnp.asarray(ybar), jnp.concatenate([jnp.asarray(ybar)[1:], jnp.asarray(ybar)[-1:]]),
          dey, rate_ey, normals)

    snap_idx = (jnp.round(jnp.asarray(snapshot_times, dtype=float) / dt).astype(int)
                if snapshot_times is not None else jnp.zeros(0, dtype=int))

    def step(state: tuple[Array, Array, Array, Array],
             inp: tuple[Array, Array, Array, Array, Array,
                        Array, Array, Array, Array, Array],
             ) -> tuple[tuple[Array, Array, Array, Array], Array]:
        x, y, z, snap = state
        i, f0, f0d, y0, dy0, yb, yb_next, de, rde, given = inp
        snap = jnp.where((snap_idx == i)[:, None, None], (f0 + x)[None], snap)
        y1 = y0 + Ad * x + c_y * y
        vol = local_vol(model, y1, yb, z)
        var = vol**2
        rho = (f0d + ekd * (x + y * Bd) - f0 - x) / delta
        mu_y = rho + var * c_y
        drift_y1 = de + rev_k * (y1 - ystar)                 # P-drift of Y_1 over the step
        lam = mu_y - (rde - a * (y1 - ystar))                 # phi^Y Lambda, (37)
        stats = jnp.stack([jnp.mean(a, axis=1) for a in (x, lam, var, y1, z)])

        if use_given:
            xi = given
        else:
            xi = jax.random.normal(jax.random.fold_in(key, i), draw_shape)
            if antithetic:
                xi = jnp.concatenate([xi, -xi], axis=2)
        z_new = variance_step(model, z, xi[1], dt)
        y_new = y * decay_y + var * gain_y                    # exact for frozen var
        noise = vol * sqdt * xi[0]
        if model.vol_scheme == "pc":
            x_pred = x + (drift_y1 - dy0 - c_y * (y_new - y)) / Ad + noise
            y1_pred = y0 + dy0 + Ad * x_pred + c_y * y_new          # = Y_1 + drift + A noise
            vol_pred = local_vol(model, y1_pred, yb_next, z_new)
            var_c, vol_c, ito = pc_vol(model, vol, vol_pred, local_vol_dx(model, y1, yb, z, Ad), dt)
            y_new = y * decay_y + var_c * gain_y
            noise = vol_c * sqdt * xi[0] + ito
        # the drift keeps E[Y_1] on its prescribed path given y_new
        x_new = x + (drift_y1 - dy0 - c_y * (y_new - y)) / Ad + noise
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

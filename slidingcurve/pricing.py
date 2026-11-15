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
from .model import Model, local_vol, local_vol_dx, pc_vol, variance_step


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


_PATH_DEVICES = 1


def set_path_devices(n: int = None):
    """Split the paths of every pricing simulation across n devices (all
    available ones if n is None; 1 switches it off).

    On a CPU, JAX sees one device unless told otherwise; start Python with
    XLA_FLAGS=--xla_force_host_platform_device_count=<cores> to get one per
    core. Each device draws its own random numbers (keys split from the
    caller's), so results change within Monte-Carlo error, but are fixed for
    a given key and device count; antithetic pairs are kept aligned (path i
    and path i + n_paths/2). Returns the previous setting."""
    global _PATH_DEVICES
    prev = _PATH_DEVICES
    _PATH_DEVICES = jax.device_count() if n is None else int(n)
    return prev


def simulate_terminal(model: Model, curve, expiry, ybar_fn, key, n_paths=2**15,
                      steps_per_year=24, measure_maturity=None, gaussian_twin=False):
    """State (x, y) at `expiry` under the forward measure of `measure_maturity`
    (the expiry itself by default).

    ybar_fn(t) is the volatility scaling Ybar_1 on an array of times. Paths
    come in antithetic pairs: path i and path i + n_paths/2.

    With gaussian_twin=True also returns (x_g, y_g, mean_g, var_g): the state
    of the b = 0, eta = 0 model with the same sigma, driven by the same
    normals, and the exact mean and variance of x_g under this discrete
    scheme (a linear Gaussian recursion), for use as a control variate.

    Paths are split across devices if set_path_devices was called.
    """
    if _PATH_DEVICES > 1:
        return _sharded(_simulate_terminal_one, model, curve, expiry, ybar_fn, key, n_paths, steps_per_year,
                        measure_maturity, gaussian_twin)
    return _simulate_terminal_one(model, curve, expiry, ybar_fn, key, n_paths, steps_per_year,
                                  measure_maturity, gaussian_twin)


def _sharded(fn, model, curve, expiry, ybar_fn, key, n_paths, steps_per_year, measure_maturity, twin):
    """Run fn on n_paths / d paths per device and reassemble [first halves, second halves]."""
    import inspect
    import numpy as _np
    from jax.sharding import Mesh, PartitionSpec as P
    try:
        from jax import shard_map
    except ImportError:                                        # older jax
        from jax.experimental.shard_map import shard_map
    d = _PATH_DEVICES
    if n_paths % (2 * d):
        raise ValueError(f"n_paths = {n_paths} must be a multiple of 2 x {d} devices")
    mesh = Mesh(_np.array(jax.devices()[:d]), ("paths",))
    nocheck = {"check_vma": False} if "check_vma" in inspect.signature(shard_map).parameters else {"check_rep": False}

    def local(k):
        out = fn(model, curve, expiry, ybar_fn, k[0], n_paths // d, steps_per_year, measure_maturity, twin)
        return jax.tree_util.tree_map(lambda a: a[None], out)

    out = shard_map(local, mesh=mesh, in_specs=P("paths"), out_specs=P("paths"), **nocheck)(
        jax.random.split(key, d))

    def merge(a):
        if a.ndim == 1:                                        # per-device scalars (twin moments): identical
            return a[0]
        h = a.shape[1] // 2
        return a.reshape(d, 2, h).transpose(1, 0, 2).reshape(-1)
    return jax.tree_util.tree_map(merge, out)


def _simulate_terminal_one(model: Model, curve, expiry, ybar_fn, key, n_paths=2**15,
                           steps_per_year=24, measure_maturity=None, gaussian_twin=False):
    """See simulate_terminal."""
    if gaussian_twin:
        if model.vol_scheme != "euler":
            raise NotImplementedError("the Gaussian twin is implemented for vol_scheme='euler' only")
        return _simulate_terminal_twin(model, curve, expiry, ybar_fn, key, n_paths, steps_per_year,
                                       measure_maturity)
    kappa, delta = model.kappa, model.delta
    n = max(int(round(expiry * steps_per_year)), 1)
    dt = expiry / n
    t = jnp.arange(n) * dt
    Bd, Ad = B(kappa, delta), A(kappa, delta)
    c_y = Bd**2 / (2.0 * delta)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2.0 * kappa * dt), B(2.0 * kappa, dt)
    half = n_paths // 2
    tau = expiry if measure_maturity is None else measure_maturity
    xs = (jnp.arange(n), curve.fwd_yield(t, t + delta), ybar_fn(t), B(kappa, tau - t),
          curve.fwd_yield(t + dt, t + dt + delta), ybar_fn(t + dt))

    def step(state, inp):
        x, y, z = state
        i, y0, yb, g, y0_next, yb_next = inp
        y1 = y0 + Ad * x + c_y * y
        vol = local_vol(model, y1, yb, z)
        var = vol**2
        xi = jax.random.normal(jax.random.fold_in(key, i), (2, half))
        xi = jnp.concatenate([xi, -xi], axis=1)
        z_new = variance_step(model, z, xi[1], dt)
        # exact for the linear part with volatility frozen over the step
        x_new = x * e1 + (y - var * g) * b1 + vol * jnp.sqrt(b2) * xi[0]
        y_new = y * e2 + var * b2
        if model.vol_scheme == "pc":
            vol_pred = local_vol(model, y0_next + Ad * x_new + c_y * y_new, yb_next, z_new)
            var_c, vol_c, ito = pc_vol(model, vol, vol_pred, local_vol_dx(model, y1, yb, z, Ad), dt)
            x_new = x * e1 + (y - var_c * g) * b1 + vol_c * jnp.sqrt(b2) * xi[0] + ito
            y_new = y * e2 + var_c * b2
        return (x_new, y_new, z_new), None

    init = (jnp.zeros(2 * half), jnp.zeros(2 * half), jnp.ones(2 * half))
    (x, y, _), _ = jax.lax.scan(step, init, xs)
    return x, y


def _simulate_terminal_twin(model, curve, expiry, ybar_fn, key, n_paths, steps_per_year,
                            measure_maturity):
    """simulate_terminal plus the Gaussian twin; same random numbers."""
    kappa, delta, sig = model.kappa, model.delta, model.sigma
    n = max(int(round(expiry * steps_per_year)), 1)
    dt = expiry / n
    t = jnp.arange(n) * dt
    Bd, Ad = B(kappa, delta), A(kappa, delta)
    c_y = Bd**2 / (2.0 * delta)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2.0 * kappa * dt), B(2.0 * kappa, dt)
    half = n_paths // 2
    tau = expiry if measure_maturity is None else measure_maturity
    xs = (jnp.arange(n), curve.fwd_yield(t, t + delta), ybar_fn(t), B(kappa, tau - t))

    def step(state, inp):
        x, y, z, xg, yg, mg, vg = state
        i, y0, yb, g = inp
        vol = local_vol(model, y0 + Ad * x + c_y * y, yb, z)
        var = vol**2
        xi = jax.random.normal(jax.random.fold_in(key, i), (2, half))
        xi = jnp.concatenate([xi, -xi], axis=1)
        x_new = x * e1 + (y - var * g) * b1 + vol * jnp.sqrt(b2) * xi[0]
        y_new = y * e2 + var * b2
        z_new = variance_step(model, z, xi[1], dt)
        xg_new = xg * e1 + (yg - sig**2 * g) * b1 + sig * jnp.sqrt(b2) * xi[0]
        mg_new = mg * e1 + (yg - sig**2 * g) * b1
        vg_new = vg * e1**2 + sig**2 * b2
        return (x_new, y_new, z_new, xg_new, yg * e2 + sig**2 * b2, mg_new, vg_new), None

    zero = jnp.zeros(2 * half)
    init = (zero, zero, jnp.ones(2 * half), zero, jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
    (x, y, _, xg, yg, mg, vg), _ = jax.lax.scan(step, init, xs)
    return x, y, (xg, yg, mg, vg)


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


def yield_option_prices(model: Model, curve, expiry, offsets, ybar_fn, key,
                        n_paths=2**15, steps_per_year=24, strike_reference="mean",
                        control_variate=False):
    """Options on the benchmark yield Y_1(T) = Y(T, T, T + delta) itself: the
    paper's "yield option" (Figures 4, 8 and 11).

    The paper does not define the contract. What its numbers pin down (see
    docs/investigation.md, A9):

    - the underlying is the continuously compounded forward yield: only it
      gives the paper's sigma = 1.1572% for an 82bp at-the-money vol (a
      simple rate needs 0.93%, a par swap rate 1.10-1.14%);
    - the measure is the (T + delta)-forward measure of the calibration
      caplets: only it reproduces the fitted vols of Figure 11 at the
      paper's stated parameters (payment at T instead is 12-14bp off);
    - the implied vol is inverted around the model's own mean of the yield
      in that measure (inverting around the curve forward gives 7-9bp
      errors against Figure 8 and prices below intrinsic).

    Strikes are `offsets` from a reference: the model mean
    (strike_reference="mean") or the curve forward Y(0, T, T + delta)
    ("forward"), which is where the paper draws its at-the-money line in
    Figures 4, 8 and 11. In the Gaussian case the smile is flat at
    A(delta) sigma sqrt(B_{2 kappa}(T) / T) and the choice is immaterial.

    Returns (mean yield, undiscounted call prices per offset, standard errors).
    Prices are E[(Y_1 - K)^+]; invert them with forward = mean and strike
    K = reference + offset (see yield_option_moneyness).

    control_variate=True subtracts the same option on the Gaussian twin
    (b = 0, eta = 0, same sigma and random numbers; see simulate_terminal)
    priced exactly, with the regression coefficient estimated per strike.
    The twin's strike sits at the same offset from its own reference, so
    in the Gaussian case the estimator has zero variance.
    """
    kappa, delta = model.kappa, model.delta
    Ad, cy = A(kappa, delta), B(kappa, delta) ** 2 / (2.0 * delta)
    fwd = curve.fwd_yield(expiry, expiry + delta)
    sim = simulate_terminal(model, curve, expiry, ybar_fn, key, n_paths, steps_per_year,
                            measure_maturity=expiry + delta, gaussian_twin=control_variate)
    x, y = sim[0], sim[1]
    y1 = fwd + Ad * x + cy * y
    mean = jnp.mean(y1)
    off = jnp.asarray(offsets)[:, None]
    ref = mean if strike_reference == "mean" else fwd
    pay = jnp.maximum(y1[None, :] - ref - off, 0.0)
    half = pay.shape[1] // 2
    pairs = 0.5 * (pay[:, :half] + pay[:, half:])
    if control_variate:
        xg, yg, mg, vg = sim[2]
        yg1 = fwd + Ad * xg + cy * yg
        mu_g, sd_g = fwd + Ad * mg + cy * yg, Ad * jnp.sqrt(vg)        # exact for the discrete scheme
        ref_g = jnp.mean(yg1) if strike_reference == "mean" else fwd
        k_g = ref_g + off[:, 0]
        exact_g = _bachelier_call(mu_g, k_g, sd_g)
        pay_g = jnp.maximum(yg1[None, :] - ref_g - off, 0.0)
        pg = 0.5 * (pay_g[:, :half] + pay_g[:, half:])
        dp, dg = pairs - pairs.mean(1, keepdims=True), pg - pg.mean(1, keepdims=True)
        beta = jnp.sum(dp * dg, 1) / jnp.maximum(jnp.sum(dg * dg, 1), 1e-300)
        pairs = pairs - beta[:, None] * (pg - exact_g[:, None])
    return mean, jnp.mean(pairs, axis=1), jnp.std(pairs, axis=1, ddof=1) / jnp.sqrt(half)


def _bachelier_call(mean, strike, sd):
    d = (mean - strike) / sd
    return (mean - strike) * jax.scipy.stats.norm.cdf(d) + sd * jax.scipy.stats.norm.pdf(d)


def yield_option_moneyness(curve, model, expiry, offsets, mean, strike_reference="mean"):
    """Strike minus forward for inverting yield_option_prices: the offsets
    themselves for the mean reference, offsets + (Y(0,T,T+delta) - mean) for
    the forward reference."""
    if strike_reference == "mean":
        return jnp.asarray(offsets)
    return jnp.asarray(offsets) + curve.fwd_yield(expiry, expiry + model.delta) - mean

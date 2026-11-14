"""Shared helpers for the investigation scripts (not part of the library).

terminal_yield simulates the benchmark yield Y_1(T) = Y(T, T, T + delta) at an
option expiry in any of the candidate measures, with the library's own
volatility function and variance scheme, so that pricing conventions can be
compared on identical dynamics:

    "P"         the sliding-curve physical measure, eq. (52)
    "Q"         the spot risk-neutral measure, eq. (41); also returns the
                pathwise discount factor exp(-int r)
    float tau   the tau-forward measure (tau = T: expiry; tau = T + delta:
                the measure of the calibration caplets)
"""
import json
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import brentq
from scipy.stats import norm

from slidingcurve.gaussian import A, B
from slidingcurve.model import local_vol, variance_step

BP = 1e4
OUT = pathlib.Path(__file__).resolve().parent / "results"


def terminal_yield(model, curve, T, ybar_fn, key, measure, n_paths=2**16, steps_per_year=24):
    """Returns (Y_1(T) per path, discount factor per path or None)."""
    kappa, delta = model.kappa, model.delta
    n = max(int(round(T * steps_per_year)), 1)
    dt = T / n
    t = jnp.arange(n) * dt
    Ad, Bd, ekd = A(kappa, delta), B(kappa, delta), jnp.exp(-kappa * delta)
    c_y = Bd**2 / (2.0 * delta)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2.0 * kappa * dt), B(2.0 * kappa, dt)
    half = n_paths // 2
    y0 = curve.fwd_yield(t, t + delta)
    dy0 = curve.fwd_yield(t + dt, t + dt + delta) - y0
    tau = T if measure in ("P", "Q") else float(measure)
    xs = (jnp.arange(n), y0, dy0, ybar_fn(t), B(kappa, tau - t), curve.forward(t))

    def step(state, inp):
        x, y, z, ir = state
        i, y0_, dy0_, yb, g, f0 = inp
        vol = local_vol(model, y0_ + Ad * x + c_y * y, yb, z)
        var = vol**2
        xi = jax.random.normal(jax.random.fold_in(key, i), (2, half))
        xi = jnp.concatenate([xi, -xi], axis=1)
        y_new = y * e2 + var * b2
        if measure == "P":
            x_new = x - (dy0_ + c_y * (y_new - y)) / Ad + vol * jnp.sqrt(dt) * xi[0]
        elif measure == "Q":
            x_new = x * e1 + y * b1 + vol * jnp.sqrt(b2) * xi[0]
        else:
            x_new = x * e1 + (y - var * g) * b1 + vol * jnp.sqrt(b2) * xi[0]
        ir = ir + 0.5 * dt * (f0 + x) + 0.5 * dt * (f0 + x_new)   # trapezoid; f0 at left point
        return (x_new, y_new, variance_step(model, z, xi[1], dt), ir), None

    z0 = jnp.zeros(2 * half)
    (x, y, _, ir), _ = jax.lax.scan(step, (z0, z0, jnp.ones(2 * half), z0), xs)
    y1 = curve.fwd_yield(T, T + delta) + Ad * x + c_y * y
    return np.asarray(y1), (np.exp(-np.asarray(ir)) if measure in ("P", "Q") else None)


def bachelier(F, K, v, T, put=False):
    s = v * np.sqrt(T)
    d = (F - K) / s
    c = (F - K) * norm.cdf(d) + s * norm.pdf(d)
    return c - (F - K) if put else c


def implied(price, F, K, T, put=False):
    """Normal implied vol; NaN where the price is at or below intrinsic (the
    points the paper's own Figure 4 leaves out)."""
    intrinsic = max(K - F, 0.0) if put else max(F - K, 0.0)
    if not np.isfinite(price) or price <= intrinsic + 1e-13:
        return np.nan
    try:
        return brentq(lambda v: bachelier(F, K, v, T, put) - price, 1e-7, 0.5, xtol=1e-14)
    except ValueError:
        return np.nan


def smile(y1, strikes, F_inv, T, weights=None, mode="call"):
    """Implied normal vols of E[w (Y - K)^+] / E[w] inverted with forward F_inv.
    mode: 'call' everywhere, or 'otm' (puts below F_inv)."""
    w = np.ones_like(y1) if weights is None else weights / weights.mean()
    out = []
    for k in strikes:
        put = mode == "otm" and k < F_inv
        pay = np.maximum(k - y1, 0.0) if put else np.maximum(y1 - k, 0.0)
        out.append(implied(float(np.mean(w * pay)), F_inv, k, T, put))
    return np.array(out)


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


# --- alternative variance schemes, for the scheme hypothesis -------------------

def euler_full_truncation(model, z, xi, dt):
    zp = jnp.maximum(z, 0.0)
    return z + model.kappa_z * (1.0 - zp) * dt + model.eta * jnp.sqrt(zp * dt) * xi


def euler_reflection(model, z, xi, dt):
    return jnp.abs(z + model.kappa_z * (1.0 - z) * dt + model.eta * jnp.sqrt(jnp.abs(z) * dt) * xi)


def use_variance_scheme(fn):
    """Swap the variance scheme everywhere it was imported by name. Returns
    the previous one. Full truncation: the library's local_vol floors z."""
    import slidingcurve.model as mdl
    import slidingcurve.pricing as prc
    import common
    prev = mdl.variance_step
    for mod in (mdl, prc, common):
        mod.variance_step = fn
    return prev


# --- alternative local-volatility functions (floor / cap / parameterisation) ---

def make_local_vol(floor="vol", cap="sigma_r", relative=False, ybar0=None):
    """floor: 'vol' (library: affine vol floored at zero), 'abs' (no floor; the
    sign of the vol is irrelevant to the dynamics so this is |vol|), 'rate'
    (benchmark yield floored at zero before the affine map).
    cap: 'sigma_r' (library: sqrt(z) * lv capped at vol_cap * sigma), 'lv'
    (local part capped, sqrt(z) uncapped), 'none'.
    relative: literal eq. (49) with constant relative vol lambda = A sigma /
    Ybar(0) (pass ybar0): sigma_r = sigma (Ybar(t) / Ybar(0)) ((1 - b) + b Y / Ybar(t))."""
    def lv_fn(model, y1, ybar, z):
        zp = jnp.maximum(z, 1e-12)
        y = jnp.maximum(y1, 0.0) if floor == "rate" else y1
        lv = model.sigma * ((1.0 - model.b) + model.b * y / ybar)
        if relative:
            lv = lv * ybar / ybar0
        lv = jnp.abs(lv) if floor == "abs" else jnp.maximum(lv, 0.0)
        c = model.vol_cap * model.sigma
        if cap == "sigma_r":
            return jnp.minimum(jnp.sqrt(zp) * lv, c)
        if cap == "lv":
            return jnp.sqrt(zp) * jnp.minimum(lv, c)
        return jnp.sqrt(zp) * lv
    return lv_fn


def use_local_vol(fn):
    import slidingcurve.model as mdl
    import slidingcurve.pricing as prc
    import common
    prev = mdl.local_vol
    for mod in (mdl, prc, common):
        mod.local_vol = fn
    return prev


def paper_curve_ext(which, ext="quadratic"):
    """markets.paper_curve with a choice of extension beyond the last plotted
    maturity: 'quadratic' (library), 'linear', 'flat' (zero yield flat), or
    'flat_fwd' (instantaneous forward flat at its value at the last node)."""
    from slidingcurve import markets
    from slidingcurve.curve import Curve
    fig = markets.paper_figures()[{"baseline": "fig2", "steep": "fig5", "market": "fig10"}[which]]
    nodes_t = markets._MARKET_NODES if which == "market" else markets._NODES
    t, y = np.asarray(fig["T"]), np.asarray(fig["Y"], dtype=float) * 1e-4
    nodes = np.interp(nodes_t, t, y)
    y0 = nodes[0] - (nodes[1] - nodes[0]) / (nodes_t[1] - nodes_t[0]) * nodes_t[0]
    T_end = nodes_t[-1]
    extra = T_end + np.array([2.5, 5.0, 10.0, 20.0])
    if ext == "quadratic":
        ev = np.polyval(np.polyfit(nodes_t[-4:], nodes[-4:], 2), extra)
    elif ext == "linear":
        ev = np.polyval(np.polyfit(nodes_t[-4:], nodes[-4:], 1), extra)
    elif ext == "flat":
        ev = np.full(extra.shape, nodes[-1])
    else:   # flat forward: T Y(T) grows linearly at the last forward
        f_end = nodes[-1] + T_end * np.polyfit(nodes_t[-4:], nodes[-4:], 1)[0]
        ev = (T_end * nodes[-1] + f_end * (extra - T_end)) / extra
    if ext == "quadratic":       # library: exactly two extra nodes
        extra, ev = extra[:2], ev[:2]
    return Curve.from_zero_nodes(np.concatenate([[0.0], nodes_t, extra]), np.concatenate([[y0], nodes, ev]))


# --- exact CIR sampling and rate dynamics driven by given variance paths -------
# (A3 and B1: compare variance schemes against exact draws on the same grid)
T, SPY = 20.0, 24
n_steps = int(T * SPY)
dt = T / n_steps


def exact_cir(eta, kz, n, seed):
    rng = np.random.default_rng(seed)
    c = eta**2 * (1 - np.exp(-kz * dt)) / (4 * kz)
    d = 4 * kz / eta**2
    z = np.ones((n_steps + 1, n))
    for i in range(n_steps):
        z[i + 1] = c * rng.noncentral_chisquare(d, z[i] * np.exp(-kz * dt) / c)
    return z


def scheme_paths(model, n, key):
    def step(z, i):
        xi = jax.random.normal(jax.random.fold_in(key, i), (n,))
        zn = variance_step(model, z, xi, dt)
        return zn, zn
    _, zs = jax.lax.scan(step, jnp.ones(n), jnp.arange(n_steps))
    return np.vstack([np.ones((1, n)), np.asarray(zs)])


def yield_from_z(model, curve, ybar_fn, zpaths, key):
    """Terminal Y_1(T) in Q^(T+delta) driven by given variance paths."""
    kappa, delta = model.kappa, model.delta
    Ad, Bd = A(kappa, delta), B(kappa, delta)
    c_y = Bd**2 / (2 * delta)
    e1, b1 = jnp.exp(-kappa * dt), B(kappa, dt)
    e2, b2 = jnp.exp(-2 * kappa * dt), B(2 * kappa, dt)
    t = jnp.arange(n_steps) * dt
    n = zpaths.shape[1]
    xs = (jnp.arange(n_steps), curve.fwd_yield(t, t + delta), ybar_fn(t), B(kappa, T + delta - t),
          jnp.asarray(zpaths[:-1]))

    def step(state, inp):
        x, y = state
        i, y0, yb, g, z = inp
        vol = local_vol(model, y0 + Ad * x + c_y * y, yb, z)
        xi = jax.random.normal(jax.random.fold_in(key, i), (n,))
        return (x * e1 + (y - vol**2 * g) * b1 + vol * jnp.sqrt(b2) * xi, y * e2 + vol**2 * b2), None

    (x, y), _ = jax.lax.scan(step, (jnp.zeros(n), jnp.zeros(n)), xs)
    return np.asarray(curve.fwd_yield(T, T + delta) + Ad * x + c_y * y)



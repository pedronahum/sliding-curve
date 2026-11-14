"""Calibration of (sigma, b, eta) to swaption quotes.

Residuals are price differences divided by the quote's normal vega, i.e.
errors in implied normal volatility to first order, in basis points. That
keeps the objective smooth and differentiable without inverting the Bachelier
formula inside the optimiser. The Monte-Carlo pricer is run on fixed random
numbers, so the objective is a deterministic function of the parameters and
its Jacobian comes from forward-mode autodiff through the simulation
(finite differences when eta is fitted; see below).
"""
from typing import NamedTuple, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares

from . import bachelier as bl, volscale
from .model import Model
from .pricing import swaption_prices

_BOUNDS = {"sigma": (1e-4, 0.05), "b": (-0.5, 1.5), "eta": (0.0, 3.0), "kappa": (0.0, 0.5)}
_SCALE = {"sigma": 0.01, "b": 1.0, "eta": 1.0, "kappa": 0.1}


class Fit(NamedTuple):
    model: Model
    vol_errors_bp: np.ndarray       # model minus market, implied normal vol
    model_vols: np.ndarray
    price_se_bp: np.ndarray         # Monte-Carlo standard error, in vol bp
    success: bool


def _solve(model, free, resid_fn):
    """Bounded least squares on a deterministic residual function of the
    free parameters (a jax function of a parameter vector).

    Residual and Jacobian come from one jitted forward-mode pass (the primal
    is computed alongside the tangents anyway), cached on the parameter
    vector, since least_squares asks for both at each accepted point.

    The quadratic-exponential variance scheme switches branch as eta moves,
    so at fixed random numbers its samples are not differentiable in eta;
    with eta free and that scheme, finite differences are used instead.
    The "lognormal" scheme is smooth in eta and keeps autodiff."""
    ad = "eta" not in free or model.variance_scheme == "lognormal"
    x0 = np.array([getattr(model, n) for n in free], dtype=float)
    lo, hi = (np.array([_BOUNDS[n][k] for n in free]) for k in (0, 1))
    kw = dict(bounds=(lo, hi), x_scale=[_SCALE[n] for n in free], xtol=1e-10)
    if not ad:
        resid = jax.jit(resid_fn)
        return least_squares(lambda th: np.asarray(resid(jnp.asarray(th))), x0, jac="2-point",
                             diff_step=5e-2, **kw)

    def with_aux(th):
        r = resid_fn(th)
        return r, r

    val_jac = jax.jit(jax.jacfwd(with_aux, has_aux=True))
    cache = {}

    def both(th):
        key = th.tobytes()
        if key not in cache:
            cache.clear()
            J, r = val_jac(jnp.asarray(th))
            cache[key] = (np.asarray(r), np.asarray(J))
        return cache[key]

    return least_squares(lambda th: both(th)[0], x0, jac=lambda th: both(th)[1], **kw)


def calibrate_yield_smile(model: Model, curve, expiry, offsets, vols, free=("sigma", "b"),
                          weights=None, key=None, n_paths=2**15, steps_per_year=24,
                          bootstrap_expiries=None, strike_reference="mean",
                          control_variate=False) -> Fit:
    """Fit to normal vols of options on the benchmark yield (see
    pricing.yield_option_prices), quoted at strike offsets from at-the-money.
    `weights` scales the residuals, e.g. a large weight on the at-the-money
    quote pins it. control_variate=True prices with the Gaussian-twin control
    variate (pricing.yield_option_prices). The volatility scaling is the unweighted bootstrap, rebuilt
    from the trial sigma inside the objective, on a monthly expiry grid
    (volscale.bootstrap_grid).

    With eta free the problem has more than one basin: a second, spurious fit
    exists at a sigma above 2.5% and eta above 1.5, where the volatility cap
    does the work. Start from a skew-only fit and a moderate eta."""
    from .pricing import yield_option_moneyness, yield_option_prices
    key = jax.random.PRNGKey(0) if key is None else key
    if "eta" in free and model.eta == 0.0:
        model = model._replace(eta=0.3)
    offsets, vols = np.asarray(offsets, dtype=float), np.asarray(vols, dtype=float)
    w = jnp.ones(len(vols)) if weights is None else jnp.asarray(weights, dtype=float)
    grid = volscale.bootstrap_grid(np.ceil(expiry)) if bootstrap_expiries is None else bootstrap_expiries
    vj = jnp.asarray(vols)

    def priced(m):
        mean, p, se = yield_option_prices(m, curve, expiry, offsets, volscale.bootstrap_fn(curve, m, grid),
                                          key, n_paths, steps_per_year, strike_reference,
                                          control_variate)
        k = yield_option_moneyness(curve, m, expiry, offsets, mean, strike_reference)
        # market prices at the same moneyness: quotes are vols at reference + offset
        return p, se, k, bl.normal_price(0.0, k, vj, expiry), bl.normal_vega(0.0, k, vj, expiry)

    def resid(theta):
        p, _, _, target, vega = priced(model._replace(**{n: theta[i] for i, n in enumerate(free)}))
        return 1e4 * w * (p - target) / vega

    sol = _solve(model, free, resid)
    final = jax.jit(lambda th: priced(model._replace(**{n: th[i] for i, n in enumerate(free)})))
    p, se, k, _, vega = final(jnp.asarray(sol.x))
    model = model._replace(**{n: float(v) for n, v in zip(free, sol.x)})
    model_vols = bl.implied_normal_vol(np.asarray(p), 0.0, np.asarray(k), expiry)
    return Fit(model, 1e4 * (model_vols - vols), model_vols, 1e4 * np.asarray(se / vega),
               bool(sol.success))


def calibrate(model: Model, curve, quotes, free: Sequence[str] = ("sigma",),
              ybar_fn=None, key=None, n_paths=2**15, steps_per_year=24) -> Fit:
    """Least-squares fit of the parameters named in `free`, starting from `model`.

    If ybar_fn is None the volatility scaling is the unweighted bootstrap on
    monthly expiries, rebuilt from the trial sigma inside the objective.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    if "eta" in free and model.eta == 0.0:
        model = model._replace(eta=0.3)     # prices are even in eta: zero gradient at 0
    grid = volscale.bootstrap_grid(np.ceil(quotes.swaption.expiry))
    scaling = (lambda m: ybar_fn) if ybar_fn is not None else \
        (lambda m: volscale.bootstrap_fn(curve, m, grid))
    swp, strikes = quotes.swaption, jnp.asarray(quotes.strikes)
    vega = quotes.annuity * np.asarray(
        bl.normal_vega(quotes.forward, strikes, jnp.asarray(quotes.normal_vols), swp.expiry))
    target, vega_j = jnp.asarray(quotes.prices), jnp.asarray(vega)

    def price(theta):
        m = model._replace(**{name: theta[i] for i, name in enumerate(free)})
        return swaption_prices(m, curve, swp, strikes, scaling(m), key, n_paths, steps_per_year)

    sol = _solve(model, free, lambda th: 1e4 * (price(th)[0] - target) / vega_j)
    p, se = jax.jit(price)(jnp.asarray(sol.x))
    fitted = model._replace(**{n: float(v) for n, v in zip(free, sol.x)})
    vols = bl.implied_normal_vol(np.asarray(p) / quotes.annuity, quotes.forward,
                                 quotes.strikes, swp.expiry)
    return Fit(fitted, 1e4 * (vols - quotes.normal_vols), vols,
               1e4 * np.asarray(se) / vega, bool(sol.success))

"""Calibration of (sigma, b, eta) to swaption quotes.

Residuals are price differences divided by the quote's normal vega, i.e.
errors in implied normal volatility to first order, in basis points. That
keeps the objective smooth and differentiable without inverting the Bachelier
formula inside the optimiser. The Monte-Carlo pricer is run on fixed random
numbers, so the objective is a deterministic function of the parameters and
its Jacobian comes from forward-mode autodiff through the simulation.
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


def calibrate(model: Model, curve, quotes, free: Sequence[str] = ("sigma",),
              ybar_fn=None, key=None, n_paths=2**15, steps_per_year=24, passes=2) -> Fit:
    """Least-squares fit of the parameters named in `free`, starting from `model`.

    If ybar_fn is None the volatility scaling is the unweighted bootstrap on
    annual expiries. It depends on sigma, so the fit is repeated `passes`
    times with the scaling rebuilt at the latest sigma.
    """
    key = jax.random.PRNGKey(0) if key is None else key
    if "eta" in free and model.eta == 0.0:
        model = model._replace(eta=0.3)     # prices are even in eta: zero gradient at 0
    if ybar_fn is None:
        grid = np.arange(1.0, np.ceil(quotes.swaption.expiry) + 1.0)
        for _ in range(passes):
            fit = calibrate(model, curve, quotes, free, volscale.bootstrap_fn(curve, model, grid),
                            key, n_paths, steps_per_year)
            model = fit.model
        return fit
    swp, strikes = quotes.swaption, jnp.asarray(quotes.strikes)
    vega = quotes.annuity * np.asarray(
        bl.normal_vega(quotes.forward, strikes, jnp.asarray(quotes.normal_vols), swp.expiry))
    target, vega_j = jnp.asarray(quotes.prices), jnp.asarray(vega)

    def price(theta):
        m = model._replace(**{name: theta[i] for i, name in enumerate(free)})
        return swaption_prices(m, curve, swp, strikes, ybar_fn, key, n_paths, steps_per_year)

    resid = jax.jit(lambda th: 1e4 * (price(th)[0] - target) / vega_j)
    jac = jax.jit(jax.jacfwd(lambda th: 1e4 * (price(th)[0] - target) / vega_j))

    x0 = np.array([getattr(model, n) for n in free], dtype=float)
    lo, hi = (np.array([_BOUNDS[n][k] for n in free]) for k in (0, 1))
    sol = least_squares(lambda th: np.asarray(resid(jnp.asarray(th))), x0,
                        jac=lambda th: np.asarray(jac(jnp.asarray(th))),
                        bounds=(lo, hi), x_scale=[_SCALE[n] for n in free], xtol=1e-10)
    fitted = model._replace(**{n: float(v) for n, v in zip(free, sol.x)})
    p, se = swaption_prices(fitted, curve, swp, strikes, ybar_fn, key, n_paths, steps_per_year)
    vols = bl.implied_normal_vol(np.asarray(p) / quotes.annuity, quotes.forward,
                                 quotes.strikes, swp.expiry)
    return Fit(fitted, 1e4 * (vols - quotes.normal_vols), vols,
               1e4 * np.asarray(se) / vega, bool(sol.success))

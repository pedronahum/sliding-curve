"""Volatility scaling Ybar_1(t) (Appendix C.3): the centre of the local
volatility, chosen so that changing the skew b leaves at-the-money
volatilities of the calibration instruments (approximately) unchanged."""
import numpy as np
import jax.numpy as jnp

from . import gaussian as g


def ybar_zeroth(curve, delta, t):
    """Zeroth order: centre on the forward, Ybar(t) = Y(0, t, t + delta)."""
    t = jnp.asarray(t, dtype=float)
    return curve.fwd_yield(t, t + delta)


def ybar_bootstrap(curve, kappa, delta, sigma, expiries, weighted=False):
    """Bootstrap (80)-(81): piecewise-constant Ybar on (T_{m-1}, T_m].

    weighted=False omits the exp(kappa t) weights, the variant Section 9.1
    says the paper's numerical results use. Returns one value per expiry.
    """
    Tm = np.asarray(expiries, dtype=float)
    lo = np.concatenate([[0.0], Tm[:-1]])
    if weighted and kappa != 0.0:
        K = (np.exp(kappa * Tm) - np.exp(kappa * lo)) / kappa
    else:
        K = Tm - lo
    ybar = np.zeros_like(Tm)
    for m in range(len(Tm)):
        # Yhat(T_k, T_m) for k <= m: benchmark yield at T_k, in the measure of
        # the option expiring at T_m.
        yh = np.asarray(g.yhat(curve, kappa, delta, sigma, jnp.asarray(Tm[: m + 1]), Tm[m]))
        ybar[m] = yh[m] + np.sum((yh[:m] - ybar[:m]) * K[:m]) / K[m]
    return ybar


def piecewise_constant(expiries, values, t):
    """Evaluate a piecewise-constant function, left-open buckets (T_{m-1}, T_m],
    flat beyond the last expiry."""
    Tm = jnp.asarray(expiries, dtype=float)
    idx = jnp.clip(jnp.searchsorted(Tm, jnp.asarray(t, dtype=float), side="left"),
                   0, Tm.shape[0] - 1)
    return jnp.asarray(values)[idx]

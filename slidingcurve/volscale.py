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
    Written in jax.numpy and differentiable in sigma, so a calibration can
    carry the dependence of the scaling on sigma inside its objective.
    """
    Tm = np.asarray(expiries, dtype=float)
    lo = np.concatenate([[0.0], Tm[:-1]])
    if weighted and kappa != 0.0:
        K = (np.exp(kappa * Tm) - np.exp(kappa * lo)) / kappa
    else:
        K = Tm - lo
    ybar = []
    for m in range(len(Tm)):
        # Yhat(T_k, T_m) for k <= m: benchmark yield at T_k, in the measure of
        # the option expiring at T_m.
        yh = g.yhat(curve, kappa, delta, sigma, jnp.asarray(Tm[: m + 1]), Tm[m])
        acc = sum((yh[k] - ybar[k]) * K[k] for k in range(m)) if m else 0.0
        ybar.append(yh[m] + acc / K[m])
    return jnp.stack(ybar)


def piecewise_constant(expiries, values, t):
    """Evaluate a piecewise-constant function, left-open buckets (T_{m-1}, T_m],
    flat beyond the last expiry."""
    Tm = jnp.asarray(expiries, dtype=float)
    idx = jnp.clip(jnp.searchsorted(Tm, jnp.asarray(t, dtype=float), side="left"),
                   0, Tm.shape[0] - 1)
    return jnp.asarray(values)[idx]


def bootstrap_fn(curve, model, expiries, weighted=False):
    """Ybar_1 as a function of time, from the bootstrap at the model's kappa,
    delta and sigma. The scaling depends on sigma: rebuild it when sigma
    changes (calibration does so inside its objective)."""
    vals = ybar_bootstrap(curve, model.kappa, model.delta, model.sigma, expiries, weighted)
    return lambda t: piecewise_constant(expiries, vals, t)

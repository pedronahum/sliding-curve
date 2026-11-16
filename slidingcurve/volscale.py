"""Volatility scaling Ybar_1(t) (Appendix C.3): the centre of the local
volatility, chosen so that changing the skew b leaves at-the-money
volatilities of the calibration instruments (approximately) unchanged."""
from collections.abc import Callable

import numpy as np
import jax.numpy as jnp

from ._typing import Array, ArrayLike, FloatSeq
from . import gaussian as g
from .curve import Curve
from .model import Model


def ybar_zeroth(curve: Curve, delta: float, t: ArrayLike) -> Array:
    """Zeroth order: centre on the forward, Ybar(t) = Y(0, t, t + delta)."""
    t = jnp.asarray(t, dtype=float)
    return curve.fwd_yield(t, t + delta)


def ybar_bootstrap(curve: Curve, kappa: float, delta: float, sigma: float,
                   expiries: ArrayLike | FloatSeq, weighted: bool = False) -> Array:
    """Bootstrap (80)-(81): piecewise-constant Ybar on (T_{m-1}, T_m].

    weighted=False omits the exp(kappa t) weights, the variant Section 9.1
    says the paper's numerical results use. Returns one value per expiry.
    Written in jax.numpy and differentiable in sigma, so a calibration can
    carry the dependence of the scaling on sigma inside its objective.

    The recursion (81),
        Ybar_m K_m = Yhat(T_m, T_m) K_m + sum_{k<m} (Yhat(T_k, T_m) - Ybar_k) K_k,
    telescopes: with R_m = sum_{k<=m} Yhat(T_k, T_m) K_k it reads
    Ybar_m K_m = R_m - R_{m-1}, so the whole scaling is one matrix of
    forward-measure means and a difference (no loop over expiries).
    """
    Tm = np.asarray(expiries, dtype=float)
    lo = np.concatenate([[0.0], Tm[:-1]])
    if weighted and kappa != 0.0:
        K = (np.exp(kappa * Tm) - np.exp(kappa * lo)) / kappa
    else:
        K = Tm - lo
    # Yhat(T_k, T_m) for all k <= m: benchmark yield at T_k, in the measure of
    # the option expiring at T_m
    tk, tm = np.meshgrid(Tm, Tm, indexing="ij")
    lower = np.triu(np.ones((len(Tm), len(Tm))))              # k <= m
    yh = g.yhat(curve, kappa, delta, sigma, jnp.asarray(np.minimum(tk, tm)), jnp.asarray(tm))
    R = jnp.sum(yh * (lower * K[:, None]), axis=0)              # R_m
    return (R - jnp.concatenate([jnp.zeros(1), R[:-1]])) / K


def piecewise_constant(expiries: ArrayLike | FloatSeq, values: ArrayLike | FloatSeq,
                       t: ArrayLike) -> Array:
    """Evaluate a piecewise-constant function, left-open buckets (T_{m-1}, T_m],
    flat beyond the last expiry."""
    Tm = jnp.asarray(expiries, dtype=float)
    idx = jnp.clip(jnp.searchsorted(Tm, jnp.asarray(t, dtype=float), side="left"),
                   0, Tm.shape[0] - 1)
    return jnp.asarray(values)[idx]


def bootstrap_grid(horizon: float | Array, per_year: int = 12) -> np.ndarray:
    """Expiry grid T_m for the bootstrap: monthly by default.

    (81) is the rectangle-rule discretisation of the continuous condition
    (80), and its error is first order in the grid spacing: on the steep
    curve at b = 1, gamma(40y) moves by 1.7, 2.6, 3.2, 3.4 and 3.4bp going
    from an annual grid to semi-annual, quarterly, monthly, fortnightly and
    weekly ones (docs/investigation.md, A2). Monthly is within 0.2bp of the
    limit."""
    return np.arange(1, int(jnp.round(per_year * horizon)) + 1) / float(per_year)


def bootstrap_fn(curve: Curve, model: Model, expiries: ArrayLike | FloatSeq,
                 weighted: bool = False) -> Callable[[ArrayLike], Array]:
    """Ybar_1 as a function of time, from the bootstrap at the model's kappa,
    delta and sigma. The scaling depends on sigma: rebuild it when sigma
    changes (calibration does so inside its objective)."""
    vals = ybar_bootstrap(curve, model.kappa, model.delta, model.sigma, expiries, weighted)
    return lambda t: piecewise_constant(expiries, vals, t)

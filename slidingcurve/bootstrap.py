"""Single-curve bootstrap from deposit and par swap quotes.

The curve is the natural cubic spline in zero yield of Curve.from_zero_nodes,
with one node per instrument maturity (and a node at T = 0 extrapolated
linearly). Because the spline is global, the node values are solved for
jointly (Newton on all quotes at once) rather than one at a time; the result
reprices every input instrument to solver tolerance.

Conventions are deliberately plain: continuous-time accruals (year
fractions are exact differences of times), one fixed-leg frequency for all
swaps, no day counts, no separate discounting curve. Real market quotes need
those conventions applied first; the bootstrap itself does not depend on them.
"""
from typing import Sequence

import numpy as np
import jax.numpy as jnp
from scipy.optimize import least_squares

from ._typing import Array
from .curve import Curve


def par_swap_rate(curve: Curve, tenor: float, frequency: int = 1, start: float = 0.0) -> Array:
    """Par rate of a swap from `start` to `start + tenor`, fixed leg paid
    `frequency` times a year: (P(start) - P(end)) / sum_i tau_i P(t_i)."""
    n = int(round(tenor * frequency))
    pay = start + jnp.arange(1, n + 1) / frequency
    ann = jnp.sum(curve.discount(pay)) / frequency
    return (curve.discount(start) - curve.discount(pay[-1])) / ann


def deposit_rate(curve: Curve, T: float) -> Array:
    """Simply compounded rate to T: (1 / P(0, T) - 1) / T."""
    return (1.0 / curve.discount(T) - 1.0) / T


def bootstrap(swap_tenors: Sequence[float], swap_rates: Sequence[float], frequency: int = 1,
              deposit_tenors: Sequence[float] = (), deposit_rates: Sequence[float] = (),
              tol: float = 1e-13) -> Curve:
    """Curve that reprices the given deposits and par swaps exactly."""
    dep_t = np.asarray(deposit_tenors, dtype=float)
    dep_r = np.asarray(deposit_rates, dtype=float)
    sw_t = np.asarray(swap_tenors, dtype=float)
    sw_r = np.asarray(swap_rates, dtype=float)
    nodes = np.concatenate([dep_t, sw_t])
    order = np.argsort(nodes)
    if np.any(np.diff(nodes[order]) <= 0):
        raise ValueError("instrument maturities must be distinct")
    quotes = np.concatenate([dep_r, sw_r])

    def build(z: np.ndarray) -> Curve:
        zs = z[order]
        x = nodes[order]
        z0 = zs[0] - (zs[1] - zs[0]) / (x[1] - x[0]) * x[0] if len(x) > 1 else zs[0]
        return Curve.from_zero_nodes(np.concatenate([[0.0], x]), np.concatenate([[z0], zs]))

    def resid(z: np.ndarray) -> np.ndarray:
        c = build(z)
        r = [float(deposit_rate(c, T)) for T in dep_t]
        r += [float(par_swap_rate(c, T, frequency)) for T in sw_t]
        # numpy's stubs make the difference Any (quotes carries dtype Any); np.asarray is
        # a no-op on an array, and it has to stay numpy because least_squares consumes it.
        return np.asarray(1e4 * (np.asarray(r) - quotes))      # basis points

    # initial guess: continuously compounded equivalents of the quotes
    z0 = np.concatenate([np.log1p(dep_r * dep_t) / np.maximum(dep_t, 1e-12),
                         frequency * np.log1p(sw_r / frequency)])
    sol = least_squares(resid, z0, xtol=tol, ftol=tol, gtol=tol, method="lm")
    if np.max(np.abs(sol.fun)) > 1e-6:
        raise RuntimeError(f"bootstrap did not converge: max error {np.max(np.abs(sol.fun)):.2e}bp")
    return build(sol.x)

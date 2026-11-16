"""Quote conventions: normal (Bachelier) and displaced-lognormal prices.

All prices are forward (undiscounted, per unit annuity) payer/call prices.
"""
from collections.abc import Callable

import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.optimize import brentq

from ._typing import Array, ArrayLike, FloatSeq

# The three `forward - strike` differences below carry `# type: ignore[operator]`.
# mypy checks every combination of two union operands, and ArrayLike contains
# both bool and np.bool_, whose __sub__ overloads do not meet; wrapping the
# operands in jnp.asarray would instead move host-side scalars onto the device.
# Every argument these functions take is genuinely array-like, so the alias
# stays. warn_unused_ignores keeps the three honest.


def normal_price(forward: ArrayLike, strike: ArrayLike, vol: ArrayLike,
                 expiry: ArrayLike) -> Array:
    """Bachelier call: E[(F_T - K)^+] with dF = vol dW."""
    s = vol * jnp.sqrt(expiry)
    d = (forward - strike) / s    # type: ignore[operator]
    return jnp.asarray((forward - strike) * norm.cdf(d)    # type: ignore[operator]
                       + s * norm.pdf(d))


def normal_vega(forward: ArrayLike, strike: ArrayLike, vol: ArrayLike,
                expiry: ArrayLike) -> Array:
    s = vol * jnp.sqrt(expiry)
    return jnp.sqrt(expiry) * norm.pdf((forward - strike) / s)    # type: ignore[operator]


def displaced_black_price(forward: ArrayLike, strike: ArrayLike, vol: ArrayLike,
                          expiry: ArrayLike, displacement: ArrayLike) -> Array:
    """Call on F where (F + displacement) is lognormal with volatility `vol`."""
    f, k = forward + displacement, strike + displacement
    s = vol * jnp.sqrt(expiry)
    d1 = jnp.log(f / k) / s + 0.5 * s
    return jnp.asarray(f * norm.cdf(d1) - k * norm.cdf(d1 - s))


def implied_normal_vol(price: ArrayLike | FloatSeq, forward: ArrayLike | FloatSeq,
                       strike: ArrayLike | FloatSeq, expiry: ArrayLike) -> np.ndarray:
    """Invert the Bachelier formula (host-side root find, for reporting)."""
    price, forward, strike = (np.asarray(a, dtype=float) for a in (price, forward, strike))
    out = np.empty(np.broadcast(price, strike).shape)
    p, f, k = np.broadcast_arrays(price, forward, strike)
    for idx in np.ndindex(out.shape):
        intrinsic = max(f[idx] - k[idx], 0.0)
        if p[idx] <= intrinsic:
            out[idx] = np.nan
            continue
        # brentq's stub wants a float-returning callable; numpy types the tuple
        # index p[idx] as an array, so the scalar is wrapped in float() (exact,
        # and already a scalar at runtime).
        g: Callable[[float], float] = (
            lambda v: float(normal_price(f[idx], k[idx], v, expiry)) - float(p[idx]))
        out[idx] = brentq(g, 1e-8, 1.0, xtol=1e-14)
    return out

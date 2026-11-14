"""Quote conventions: normal (Bachelier) and displaced-lognormal prices.

All prices are forward (undiscounted, per unit annuity) payer/call prices.
"""
import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import norm
from scipy.optimize import brentq


def normal_price(forward, strike, vol, expiry):
    """Bachelier call: E[(F_T - K)^+] with dF = vol dW."""
    s = vol * jnp.sqrt(expiry)
    d = (forward - strike) / s
    return (forward - strike) * norm.cdf(d) + s * norm.pdf(d)


def normal_vega(forward, strike, vol, expiry):
    s = vol * jnp.sqrt(expiry)
    return jnp.sqrt(expiry) * norm.pdf((forward - strike) / s)


def displaced_black_price(forward, strike, vol, expiry, displacement):
    """Call on F where (F + displacement) is lognormal with volatility `vol`."""
    f, k = forward + displacement, strike + displacement
    s = vol * jnp.sqrt(expiry)
    d1 = jnp.log(f / k) / s + 0.5 * s
    return f * norm.cdf(d1) - k * norm.cdf(d1 - s)


def implied_normal_vol(price, forward, strike, expiry):
    """Invert the Bachelier formula (host-side root find, for reporting)."""
    price, forward, strike = (np.asarray(a, dtype=float) for a in (price, forward, strike))
    out = np.empty(np.broadcast(price, strike).shape)
    p, f, k = np.broadcast_arrays(price, forward, strike)
    for idx in np.ndindex(out.shape):
        intrinsic = max(f[idx] - k[idx], 0.0)
        if p[idx] <= intrinsic:
            out[idx] = np.nan
            continue
        g = lambda v: float(normal_price(f[idx], k[idx], v, expiry)) - p[idx]
        out[idx] = brentq(g, 1e-8, 1.0, xtol=1e-14)
    return out

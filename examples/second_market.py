"""A second smile market: the USD 10y x 10y swaption smile of 14 March 2017,
from Choi, Liu and Seo (arXiv:1809.04035; see markets.choi_2017_10y10y).

The source gives three prices per unit annuity and the forward swap rate,
not the discount curve or the swap conventions, so this example assumes a
flat continuously compounded curve chosen to reproduce the forward swap
rate, with a semi-annual fixed leg (the USD convention), and says so.
It fits (sigma, b) and (sigma, b, eta) with the swaption pricer and reports
the convexity the fits imply on that flat curve.

    python examples/second_market.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from slidingcurve import (Curve, Model, bachelier as bl, calibration, decompose, markets, pricing as pr,
                          simulate, time_grid, volscale)

BP = 1e4


def main():
    expiry, fwd, offsets, prices, vols = markets.choi_2017_10y10y()
    swp = pr.Swaption.regular(expiry, 10.0, frequency=2)
    flat = brentq(lambda r: float(pr.forward_swap_rate(Curve.flat(r), swp)) - fwd, 0.0, 0.1, xtol=1e-14)
    curve = Curve.flat(flat)
    ann = float(pr.annuity(curve, swp))
    strikes = fwd + offsets
    q = markets.Quotes(swp, fwd, ann, strikes, ann * prices, vols)
    print(markets.CHOI_2017_SOURCE)
    print(f"assumed curve: flat {100*flat:.4f}% (continuous), semi-annual fixed leg; annuity {ann:.4f}")
    print("market normal vols (bp) at -100, 0, +100: " + " ".join(f"{BP*v:.2f}" for v in vols))
    base = Model(kappa=0.03, delta=5.0, sigma=0.01, variance_scheme="lognormal")
    for free in (("sigma",), ("sigma", "b"), ("sigma", "b", "eta")):
        start = base._replace(eta=0.3) if "eta" in free else base
        f = calibration.calibrate(start, curve, q, free=free, n_paths=2**15)
        m = f.model
        t = time_grid(30.0)
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(30.0))(t)
        d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**15), m, curve,
                      jnp.array([10.0, 20.0, 30.0]))
        print(f"fit {'+'.join(free):14s}: sigma {100*m.sigma:.3f}% b {m.b:+.3f} eta {m.eta:.3f} | errors "
              + " ".join(f"{e:+.2f}" for e in f.vol_errors_bp) + " bp | gamma(10, 20, 30y) "
              + " ".join(f"{BP*g:.1f}" for g in d.gamma) + " bp")


if __name__ == "__main__":
    main()

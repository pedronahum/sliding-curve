"""A6/A10: the skew fit on the Section 9.4 smile.

1. No fitting: the smile at the paper's stated parameters (sigma = 1.1999%,
   b = 0.367, and the text's 1.12%) under each strike reference, against the
   paper's own fitted vols plotted in Figure 11.
2. The fit (sigma, b) under each strike reference and weighting, with
   gamma(30y) of Figure 12 (paper 140.3bp) at the fitted parameters.
"""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import (Model, bachelier as bl, calibration, decompose, markets, pricing as pr,
                          simulate, time_grid, volscale)

N = 2**16
curve = markets.paper_curve("market")
offsets, market, paper_fit = markets.paper_smile()
base = Model(kappa=0.03, delta=5.0, sigma=0.01)
grid = np.arange(1.0, 21.0)
key = jax.random.PRNGKey(0)
fig12 = markets.paper_figures()["fig12"]
print("market quotes (bp):      ", np.round(BP * market, 2))
print("paper's fitted vols (bp):", np.round(BP * paper_fit, 2))


def smile_at(m, ref):
    mean, p, se = pr.yield_option_prices(m, curve, 20.0, offsets, volscale.bootstrap_fn(curve, m, grid),
                                         key, N, 24, ref)
    k = pr.yield_option_moneyness(curve, m, 20.0, offsets, mean, ref)
    v = bl.implied_normal_vol(np.asarray(p), 0.0, np.asarray(k), 20.0)
    return BP * v, BP * np.asarray(se / bl.normal_vega(0.0, k, v, 20.0))


def gamma30(m):
    t = time_grid(30.0)
    d = decompose(simulate(m, curve, t, volscale.bootstrap_fn(curve, m, np.arange(1.0, 31.0))(t),
                           jax.random.PRNGKey(1), n_paths=N), m, curve, jnp.array([30.0]))
    return BP * float(d.gamma[0]), BP * float(d.gamma_se[0])


out = {"stated": {}, "fits": {}}
print("\n1. smile at stated parameters vs the paper's fitted vols")
for sig in (0.011999, 0.0112):
    for ref in ("mean", "forward"):
        v, se = smile_at(base._replace(sigma=sig, b=0.367), ref)
        err = v - BP * paper_fit
        out["stated"][f"{sig}|{ref}"] = {"vols": v.tolist(), "se": se.tolist(), "err_vs_paper_fit": err.tolist()}
        print(f"  sigma {100*sig:.4f}%, strikes at {ref:7s}: {np.round(v,2)}  minus paper fit {np.round(err,2)}"
              f"  (s.e. {se.max():.2f})")

print("\n2. fits")
WEIGHTS = {"equal": None, "ATM pinned (x30)": [1, 1, 30, 1, 1]}
for ref in ("mean", "forward"):
    for wname, w in WEIGHTS.items():
        f = calibration.calibrate_yield_smile(base, curve, 20.0, offsets, market, free=("sigma", "b"),
                                              weights=w, key=key, n_paths=N, strike_reference=ref)
        g, gse = gamma30(f.model)
        out["fits"][f"{ref}|{wname}"] = {"sigma": f.model.sigma, "b": f.model.b, "vols": (BP * f.model_vols).tolist(),
                                         "gamma30": g, "gamma30_se": gse}
        print(f"  {ref:7s} {wname:17s}: sigma {100*f.model.sigma:.4f}% b {f.model.b:.4f} | vols "
              f"{np.round(BP*f.model_vols,1)} | gamma(30y) {g:.1f} +- {gse:.1f} (paper 140.3)", flush=True)
for sig, b in ((0.011999, 0.367),):
    g, gse = gamma30(base._replace(sigma=sig, b=b))
    out["gamma30_stated"] = [g, gse]
    print(f"  gamma(30y) at the paper's stated parameters: {g:.1f} +- {gse:.1f}")
save("a6_skew_fit", out)

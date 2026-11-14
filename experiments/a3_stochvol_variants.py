"""A3/A4: Figure 9 (convexity at fixed ATM vol) under each hypothesis.

Baseline curve, b = 0, kappa_z = 0.1, sigma_0 = 1%. For every variant: the
20y x 5y ATM yield-option vol at eta = 0 and at fixed sigma, sigma
recalibrated at each eta to the eta = 0 ATM vol, and gamma(40y). The
library's choices are the 'library' row. Recalibration noise is measured by
repeating the library row over pricing seeds.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
import common
from common import BP, save
from slidingcurve import (Model, bachelier as bl, calibration, decompose, markets,
                          pricing as pr, simulate, time_grid, volscale)

N = 2**16
curve = markets.paper_curve("baseline")
figs = markets.paper_figures()
base = Model(kappa=0.03, delta=5.0, sigma=0.01, kappa_z=0.1)
ybar = lambda t: volscale.ybar_zeroth(curve, 5.0, t)      # unused at b = 0
mats = jnp.array([10.0, 20.0, 30.0, 40.0])
PAPER9 = {e: figs["fig9"][f"eta={e}"][-1] for e in (0.0, 0.5, 1.0, 1.5)}
PAPER8 = {e: figs["fig8"][f"eta={e}"][5] for e in (0.0, 0.5, 1.0, 1.5)}


def atm(model, key, spy):
    _, p, se = pr.yield_option_prices(model, curve, 20.0, np.array([0.0]), ybar, key, N, spy)
    v = bl.implied_normal_vol(np.asarray(p), 0.0, np.array([0.0]), 20.0)[0]
    return v, float(se[0] / bl.normal_vega(0.0, 0.0, v, 20.0))


def gamma40(model, key, spy):
    t = time_grid(40.0, steps_per_year=spy)
    d = decompose(simulate(model, curve, t, ybar(t), key, n_paths=N), model, curve, mats)
    return BP * float(d.gamma[-1]), BP * float(d.gamma_se[-1])


def run(label, price_cap=10.0, decomp_cap=10.0, spy_price=24, spy_decomp=12, target=None,
        etas=(0.5, 1.0, 1.5), pkey=0, skey=1):
    pk, sk = jax.random.PRNGKey(pkey), jax.random.PRNGKey(skey)
    v0, _ = atm(base._replace(vol_cap=price_cap), pk, spy_price)
    tgt = v0 if target is None else target
    row = {"atm_eta0": BP * v0}
    for eta in etas:
        fixed = base._replace(eta=eta, vol_cap=price_cap)
        vfix, vse = atm(fixed, pk, spy_price)
        fit = calibration.calibrate_yield_smile(fixed, curve, 20.0, np.array([0.0]), [tgt],
                                                free=("sigma",), key=pk, n_paths=N,
                                                steps_per_year=spy_price)
        s = fit.model.sigma
        g, gse = gamma40(base._replace(eta=eta, sigma=s, vol_cap=decomp_cap), sk, spy_decomp)
        row[eta] = {"atm_fixed": BP * vfix, "atm_se": BP * vse, "sigma": s, "gamma40": g, "gamma_se": gse}
    print(f"{label:34s} ATM0 {row['atm_eta0']:6.2f} | " + " | ".join(
        f"eta {e}: ATM {row[e]['atm_fixed']:5.2f}+-{row[e]['atm_se']:.2f} sig {100*row[e]['sigma']:.4f}% "
        f"g40 {row[e]['gamma40']:6.1f}+-{row[e]['gamma_se']:.1f} ({PAPER9[e]:.1f})" for e in etas), flush=True)
    return row


out = {}
print("paper: ATM", PAPER8, " gamma40", PAPER9)
out["library"] = run("library (QE, cap 10x both, 24/12)")
for s in (1, 2, 3):
    out[f"library seed {s}"] = run(f"library, pricing seed {s}", pkey=10 + s, skey=20 + s)
out["target 71.5"] = run("recalibrate to 71.5bp (paper's text)", target=71.5e-4)
out["no cap"] = run("no cap anywhere", price_cap=1e9, decomp_cap=1e9)
out["cap pricing only"] = run("cap in pricing only", decomp_cap=1e9)
out["cap decomp only"] = run("cap in decomposition only", price_cap=1e9)
out["cap 5x"] = run("cap 5x", price_cap=5.0, decomp_cap=5.0)
out["cap 20x"] = run("cap 20x", price_cap=20.0, decomp_cap=20.0)
out["steps 52/52"] = run("52 steps a year in both", spy_price=52, spy_decomp=52)
out["steps 12/12"] = run("12 steps a year in both", spy_price=12, spy_decomp=12)
prev = common.use_variance_scheme(common.euler_full_truncation)
out["euler FT"] = run("full-truncation Euler")
common.use_variance_scheme(common.euler_reflection)
out["euler reflect"] = run("reflection Euler")
common.use_variance_scheme(prev)
save("a3_stochvol_variants", out)

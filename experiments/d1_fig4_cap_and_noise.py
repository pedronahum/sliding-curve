"""D1: two untested mechanisms for Figure 4's shape.

S1 (the b = 0 droop and its missing wing points) and S3 (the b = 1 slope of
110bp at 20y against this code's 25bp) need, per A1, forward mismatches of
opposite sign, so no single pricing convention gives both. These two tests
look elsewhere:

T1  the volatility cap. local_vol clips sigma_r at vol_cap * sigma. At b >= 0.5
    the cap binds in the right tail, where the skew lives: capping removes
    exactly the volatility that makes the smile steep, and (A2) it also caps
    the convexity. If the paper's pricing runs carried no cap (or a much
    looser one) while its decomposition runs carried the 10x cap, Figure 4's
    steep skews and Figures 3/6's convexity are both explained, and Figure 8
    (b = 0, where the cap never binds) is untouched. Measured: the 20y b = 1
    smile slope between -80 and +200bp (paper 109bp, library 25bp) and the
    ATM change b = 1 vs b = 0 (paper +5.9bp) for vol_cap = 10, 20, 50, 1e6.

T2  sampling noise in the wings. implied_normal_vol returns NaN at or below
    intrinsic, and sigma_imp(P) is concave in P in the right wing, so a noisy
    price biases the wing vol down and a price at or below intrinsic drops
    the point. Measured: the b = 0 20y smile at 4,096 and 16,384 pseudo-random
    paths over 64 seeds: the droop at +120bp (paper -13.3bp) and how often
    +160 and +200bp are absent.

    python experiments/d1_fig4_cap_and_noise.py
"""
import json
import pathlib

import jax
import numpy as np

from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale

BP = 1e4
OUT = pathlib.Path(__file__).parent / "results"


def smile(model, curve, expiry, offsets, ybar_fn, key, n_paths):
    _, p, _ = pr.yield_option_prices(model, curve, expiry, offsets, ybar_fn, key,
                                     n_paths=n_paths)
    return BP * bl.implied_normal_vol(np.asarray(p), 0.0, offsets, expiry)


def slope(vols, offsets):
    """S3: vol(+200) - vol(-80), in bp, ignoring missing points."""
    i_lo = int(np.argmin(np.abs(offsets / BP + 80.0)))
    i_hi = int(np.argmin(np.abs(offsets / BP - 200.0)))
    if not np.isfinite(vols[i_lo]) or not np.isfinite(vols[i_hi]):
        return None
    return vols[i_hi] - vols[i_lo]


def main():
    curve = markets.paper_curve("baseline")
    ref = markets.paper_figures()["fig4"]
    offsets = 1e-4 * np.asarray(ref["20"]["offsets"])
    grid = volscale.bootstrap_grid(20.0)
    res = {"t1_cap": [], "t2_noise": []}

    print("T1: 20y smile, vol_cap sweep (paper: b=1 slope 109, ATM change +5.9)")
    print(" vol_cap |  b=1 slope |  b=0 ATM |  b=1 ATM |  ATM change")
    for cap in (10.0, 20.0, 50.0, 1e6):
        row = {"vol_cap": cap}
        vols = {}
        for b in (0.0, 1.0):
            m = Model(kappa=0.03, delta=5.0, sigma=0.01, b=b, vol_cap=cap)
            vols[b] = smile(m, curve, 20.0, offsets, volscale.bootstrap_fn(curve, m, grid),
                            jax.random.PRNGKey(0), 2**16)
        row["b1_slope"] = slope(vols[1.0], offsets)
        row["b0_atm"] = float(vols[0.0][5])
        row["b1_atm"] = float(vols[1.0][5])
        row["atm_change"] = row["b1_atm"] - row["b0_atm"]
        row["b0_vols"] = [None if not np.isfinite(v) else round(float(v), 2) for v in vols[0.0]]
        row["b1_vols"] = [None if not np.isfinite(v) else round(float(v), 2) for v in vols[1.0]]
        res["t1_cap"].append(row)
        print(f" {cap:7.0f} | {row['b1_slope']:10.1f} | {row['b0_atm']:8.2f} | "
              f"{row['b1_atm']:8.2f} | {row['atm_change']:+9.2f}")

    print("\nT2: b = 0 20y smile, wing noise (paper: droop at +120 = -13.3, "
          "+160/+200 absent)")
    m0 = Model(kappa=0.03, delta=5.0, sigma=0.01)
    ybar0 = volscale.bootstrap_fn(curve, m0, grid)
    for n_paths in (2**12, 2**14):
        droops, missing = [], {"160": 0, "200": 0}
        for seed in range(64):
            v = smile(m0, curve, 20.0, offsets, ybar0, jax.random.PRNGKey(seed), n_paths)
            droops.append(float(v[8] - v[5]))
            for off, key in ((160.0, "160"), (200.0, "200")):
                i = int(np.argmin(np.abs(offsets / BP - off)))
                missing[key] += int(not np.isfinite(v[i]))
        droops = np.array(droops)
        row = {"n_paths": n_paths, "droop_mean": float(droops.mean()),
               "droop_sd": float(droops.std(ddof=1)),
               "droop_min": float(droops.min()),
               "frac_below_minus8": float((droops <= -8).mean()),
               "missing_160": missing["160"], "missing_200": missing["200"]}
        res["t2_noise"].append(row)
        print(f" {n_paths:6d} paths | droop at +120: {row['droop_mean']:+6.2f} +- "
              f"{row['droop_sd']:.2f} (min {row['droop_min']:+.2f}), <= -8bp in "
              f"{row['frac_below_minus8']:.0%} of seeds | missing: +160 "
              f"{missing['160']}/64, +200 {missing['200']}/64")

    (OUT / "d1_fig4_cap_and_noise.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'd1_fig4_cap_and_noise.json'}")


if __name__ == "__main__":
    main()

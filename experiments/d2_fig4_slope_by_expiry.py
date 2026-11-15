"""D2: Figure 4's excess skew as a function of expiry.

D1 refuted the cap and the wing noise as sources of Figure 4's shape. This
records the shape of the discrepancy itself, for Q1: the slope of the b = 1
smile between -80 and +200bp and the at-the-money change b = 1 vs b = 0, at
the four expiries of Figure 4, library against the digitised figure. Any
explanation of Figure 4 has to reproduce how the excess grows with expiry.

    python experiments/d2_fig4_slope_by_expiry.py
"""
import json
import pathlib

import jax
import numpy as np

from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale

BP = 1e4
OUT = pathlib.Path(__file__).parent / "results"


def vols(model, curve, expiry, offsets, key):
    _, p, _ = pr.yield_option_prices(model, curve, expiry, offsets,
                                     volscale.bootstrap_fn(curve, model, volscale.bootstrap_grid(20.0)),
                                     key, n_paths=2**16)
    return BP * bl.implied_normal_vol(np.asarray(p), 0.0, offsets, expiry)


def main():
    curve = markets.paper_curve("baseline")
    fig4 = markets.paper_figures()["fig4"]
    res = []
    print("expiry | slope(-80 -> +200): library / paper / ratio | ATM change b=1-b=0: "
          "library / paper")
    for expiry in (5.0, 10.0, 15.0, 20.0):
        offsets = 1e-4 * np.asarray(fig4[str(int(expiry))]["offsets"])
        row = {"expiry": expiry}
        for b in (0.0, 1.0):
            v = vols(Model(kappa=0.03, delta=5.0, sigma=0.01, b=b), curve, expiry,
                     offsets, jax.random.PRNGKey(0))
            paper = np.array(fig4[str(int(expiry))][f"b={b:.2f}"], dtype=float)
            sl = lambda w: w[10] - w[3]
            row[f"lib_b{b}"] = [None if not np.isfinite(x) else round(float(x), 2) for x in v]
            row[f"paper_b{b}"] = [None if not np.isfinite(x) else round(float(x), 2) for x in paper]
            row[f"slope_b{b}"] = (float(sl(v)), float(sl(paper)))
            row[f"atm_b{b}"] = (float(v[5]), float(paper[5]))
        ratio = row["slope_b1.0"][1] / row["slope_b1.0"][0]
        row["slope_ratio"] = ratio
        row["atm_change"] = (row["atm_b1.0"][0] - row["atm_b0.0"][0],
                             row["atm_b1.0"][1] - row["atm_b0.0"][1])
        res.append(row)
        print(f" {expiry:4.0f}y | {row['slope_b1.0'][0]:6.1f} / {row['slope_b1.0'][1]:6.1f} / "
              f"{ratio:5.2f} | {row['atm_change'][0]:+6.2f} / {row['atm_change'][1]:+6.2f}")
    (OUT / "d2_fig4_slope_by_expiry.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'd2_fig4_slope_by_expiry.json'}")


if __name__ == "__main__":
    main()

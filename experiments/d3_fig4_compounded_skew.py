"""D3: is Figure 4 the model with the skew compounded per benchmark period?

D2 measured the excess skew of Figure 4 against this code: the slope of the
b = 1 smile between -80 and +200bp is 1.01, 2.00, 3.01 and 4.28 times this
code's at 5, 10, 15 and 20 years, while the at-the-money levels and the 5y
panel agree. A slope ratio of exactly T / delta with the level preserved is
what a local volatility sigma ((1 - b) + b Y1 / Ybar) raised to an effective
exponent b_eff = b T / delta gives to first order: the skew applied once per
benchmark period and compounded, instead of applied as a level.

Test: price the Figure 4 smiles with b_eff = b T / delta in three local-vol
forms that all reduce to the library's at b_eff = b and to 1 at b = 0,

    affine  (1 - beff) + beff R          (the library's, with b -> beff)
    power   R ** beff
    exp     exp(beff (R - 1))

with R = Y1 / Ybar, and compare every panel and skew against the digitised
figure: the slope, the at-the-money change b vs b = 0, and the mean absolute
error over the paper's finite points. The b = 0 lines are untouched by all
three forms (beff = 0), so they test only the skewed lines.

    python experiments/d3_fig4_compounded_skew.py
"""
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale

BP = 1e4
OUT = pathlib.Path(__file__).parent / "results"
FORMS = ("library", "affine", "power")


def patched_local_vol(form, beff):
    def f(model, y1, ybar, z):
        zp = jnp.maximum(z, 1e-12)
        r = y1 / ybar
        if form == "affine":
            g = (1.0 - beff) + beff * r
        elif form == "power":
            g = r ** beff
        else:
            g = jnp.exp(beff * (r - 1.0))
        return jnp.clip(jnp.sqrt(zp) * model.sigma * g, 0.0, model.vol_cap * model.sigma)
    return f


def invert(p, offsets, expiry):
    """implied_normal_vol point by point: a form whose feedback explodes can
    price a wing above the library's 100% brentq bracket, which is a datum
    (that form is refuted), not an error to propagate."""
    out = []
    for pi, k in zip(np.asarray(p), offsets):
        try:
            out.append(float(bl.implied_normal_vol(np.array(pi), 0.0, np.array(k), expiry)))
        except ValueError:
            out.append(np.nan)
    return np.array(out)


def smile_vols(form, beff, b, curve, expiry, offsets, key, n_paths):
    m = Model(kappa=0.03, delta=5.0, sigma=0.01, b=b if form == "library" else 0.0,
              vol_cap=1e6)
    old = pr.local_vol
    pr.local_vol = patched_local_vol(form, beff) if form != "library" else old
    try:
        _, p, _ = pr.yield_option_prices(m, curve, expiry, offsets,
                                         volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(20.0)),
                                         key, n_paths=n_paths)
    finally:
        pr.local_vol = old
    return BP * invert(p, offsets, expiry)


def main():
    curve = markets.paper_curve("baseline")
    fig4 = markets.paper_figures()["fig4"]
    res = []
    for expiry in (5.0, 10.0, 20.0):
        offsets = 1e-4 * np.asarray(fig4[str(int(expiry))]["offsets"])
        for b in (0.5, 1.0):
            paper = np.array(fig4[str(int(expiry))][f"b={b:.2f}"], dtype=float)
            beff = b * expiry / 5.0
            for form in FORMS:
                v = smile_vols(form, beff if form != "library" else b, b, curve, expiry,
                               offsets, jax.random.PRNGKey(0), 2**15)
                fin = np.isfinite(paper) & np.isfinite(v)
                row = {"expiry": expiry, "b": b, "form": form, "beff": beff,
                       "slope": float(v[10] - v[3]), "slope_paper": float(paper[10] - paper[3]),
                       "atm": float(v[5]), "atm_paper": float(paper[5]),
                       "mae": float(np.nanmean(np.abs(v - paper)))}
                res.append(row)
                print(f"{expiry:4.0f}y b={b:.2f} {form:8s} b_eff={beff:4.1f} | slope "
                      f"{row['slope']:6.1f} (paper {row['slope_paper']:6.1f}) | ATM {row['atm']:6.2f} "
                      f"(paper {row['atm_paper']:6.2f}) | mae {row['mae']:5.2f}bp")
    (OUT / "d3_fig4_compounded_skew.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT / 'd3_fig4_compounded_skew.json'}")


if __name__ == "__main__":
    main()

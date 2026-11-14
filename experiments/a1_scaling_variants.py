"""A1: Figure 4 under the volatility-scaling and local-vol hypotheses, one at
a time, with strikes referenced to the model mean (library) and to the curve
forward. Reports the mean abs error per panel, the 20y b = 1 smile slope
(vol at +200 minus vol at -80, bp) and the change of ATM vol from b = 0 to
b = 1 at 5y and 20y (paper's text: about -4bp and -6 to -8bp; paper's
figure: about 0 and +6bp)."""
import numpy as np
import jax
import common
from common import BP, save
from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale

N = 2**15
curve = markets.paper_curve("baseline")
fig4 = markets.paper_figures()["fig4"]
base = Model(kappa=0.03, delta=5.0, sigma=0.01)
key = jax.random.PRNGKey(0)
SKEWS = (0.0, 0.25, 0.5, 0.75, 1.0)


def scaling(m, kind):
    if kind == "zeroth":
        return lambda t: volscale.ybar_zeroth(curve, 5.0, t)
    grid = {"annual": np.arange(1.0, 21.0), "monthly": np.arange(1, 241) / 12.0}[kind.split("_")[0]]
    return volscale.bootstrap_fn(curve, m, grid, weighted=kind.endswith("_w"))


def run(label, kind="annual_u", ref="mean"):
    res, errs = {}, []
    for T in (5, 10, 15, 20):
        r = fig4[str(T)]
        off = 1e-4 * np.asarray(r["offsets"])
        for b in SKEWS:
            m = base._replace(b=b)
            mean, p, _ = pr.yield_option_prices(m, curve, float(T), off, scaling(m, kind), key, N, 24, ref)
            k = pr.yield_option_moneyness(curve, m, float(T), off, mean, ref)
            v = BP * bl.implied_normal_vol(np.asarray(p), 0.0, np.asarray(k), float(T))
            res[T, b] = v
            paper = np.array(r[f"b={b:.2f}"], dtype=float)
            ok = ~np.isnan(v) & ~np.isnan(paper)
            errs.append((T, np.abs(v - paper)[ok]))
    per = {T: float(np.mean(np.concatenate([e for t, e in errs if t == T]))) for T in (5, 10, 15, 20)}
    slope = res[20, 1.0][10] - res[20, 1.0][3]
    d5, d20 = res[5, 1.0][5] - res[5, 0.0][5], res[20, 1.0][5] - res[20, 0.0][5]
    print(f"{label:38s} err 5/10/15/20y {per[5]:4.1f} {per[10]:4.1f} {per[15]:4.1f} {per[20]:4.1f} | 20y b=1 slope "
          f"{slope:5.1f} (paper 110) | ATM b1-b0: 5y {d5:+5.1f}, 20y {d20:+5.1f}", flush=True)
    return {"err": per, "slope20": float(slope), "datm5": float(d5), "datm20": float(d20),
            "smiles": {f"{T}|{b}": v.tolist() for (T, b), v in res.items()}}


import sys
SKIP_DONE = "--rest" in sys.argv
out = {}
for ref in (() if SKIP_DONE else ("mean", "forward")):
    out[f"library|{ref}"] = run(f"library, strikes at {ref}", ref=ref)
for kind in (("monthly_u", "annual_w") if SKIP_DONE else ("zeroth", "monthly_u", "annual_w")):
    out[kind] = run(f"scaling {kind}", kind=kind)
for lab, kw in (("no floor (|vol|)", dict(floor="abs")), ("floor on rate", dict(floor="rate"))):
    prev = common.use_local_vol(common.make_local_vol(**kw)); out[lab] = run(lab); common.use_local_vol(prev)
y0 = float(curve.fwd_yield(0.0, 5.0))
prev = common.use_local_vol(common.make_local_vol(relative=True, ybar0=y0))
out["relative vol flat"] = run("relative vol flat (literal eq. 49)")
common.use_local_vol(prev)
save("a1_scaling_variants" + ("_rest" if SKIP_DONE else ""), out)

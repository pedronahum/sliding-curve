"""A2: convexity at high skew (Figures 3 and 6) under each hypothesis, one
at a time, with the effect on what already matches (b = 0.25 on both curves,
Figure 12 at the paper's sigma = 1.1999%, b = 0.367)."""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
import common
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

N = 2**16
figs = markets.paper_figures()
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)


def ref(fig, series, T):
    t = np.asarray(figs[fig]["T"]); return figs[fig][series][int(np.argmin(np.abs(t - T)))]


TARGETS = [("baseline", "fig3", 1.0, 40), ("baseline", "fig3", 1.0, 20), ("baseline", "fig3", 0.75, 40),
           ("steep", "fig6", 1.0, 40), ("steep", "fig6", 0.75, 40),
           ("baseline", "fig3", 0.25, 40), ("steep", "fig6", 0.25, 40), ("market", "fig12", 0.367, 30)]


def scaling(curve, m, kind, t):
    if kind == "zeroth":
        return volscale.ybar_zeroth(curve, m.delta, t)
    if kind.startswith("boot"):
        _, step, wtd = kind.split("_")
        grid = np.arange(1.0, 41.0) if step == "annual" else np.arange(1, 481) / 12.0
        return volscale.bootstrap_fn(curve, m, grid, weighted=(wtd == "w"))(t)
    raise ValueError(kind)


def run(label, scale="boot_annual_u", spy=12, ext="quadratic", seed=1):
    t0 = time.time()
    row = {}
    for which, fig, b, T in TARGETS:
        curve = common.paper_curve_ext(which, ext)
        m = BASE._replace(b=b, sigma=0.011999 if which == "market" else 0.01)
        horizon = 30.0 if which == "market" else 40.0
        t = time_grid(horizon, steps_per_year=spy)
        ybar = scaling(curve, m, scale, t)
        d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(seed), n_paths=N), m, curve,
                      jnp.array([float(T)]))
        key = f"{which} b={b} T={T}"
        paper = ref(fig, "gamma" if fig == "fig12" else f"b={b:.2f}", T)
        row[key] = (BP * float(d.gamma[0]), BP * float(d.gamma_se[0]), paper)
    print(f"{label:30s} " + " | ".join(f"{k.replace('baseline','base').replace('market','mkt')}: "
                                       f"{v[0]:6.1f}+-{v[1]:.1f} ({v[0]-v[2]:+5.1f})" for k, v in row.items())
          + f"  [{time.time() - t0:.0f}s]", flush=True)
    return row


out = {}
ONLY = sys.argv[1:]
print("each cell: gamma (bp) +- s.e. (model minus paper)")
if not ONLY or "library" in ONLY:
    out["library"] = run("library")
    out["seed 2"] = run("library, seed 2", seed=2)
if not ONLY or "zeroth" in ONLY:
    out["zeroth"] = run("zeroth-order scaling", scale="zeroth")
out["boot monthly"] = run("bootstrap, monthly grid", scale="boot_monthly_u")
out["boot weighted"] = run("bootstrap, weighted (80)", scale="boot_annual_w")
out["steps 24"] = run("24 steps a year", spy=24)
out["steps 52"] = run("52 steps a year", spy=52)
for ext in ("linear", "flat", "flat_fwd"):
    out[f"ext {ext}"] = run(f"extension beyond grid: {ext}", ext=ext)
for lab, kw in (("floor abs (no floor)", dict(floor="abs")), ("floor on rate", dict(floor="rate")),
                ("cap on local part only", dict(cap="lv")), ("no cap", dict(cap="none"))):
    prev = common.use_local_vol(common.make_local_vol(**kw))
    out[lab] = run(lab)
    common.use_local_vol(prev)

save("a2_skew_variants" + ("_" + "_".join(ONLY) if ONLY else ""), out)

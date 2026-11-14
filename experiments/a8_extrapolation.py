"""A8: sensitivity of the decomposition at the end of the plotted range to
the curve beyond it. Market curve (plotted to 30y; Figure 10, b = 0,
sigma = 1.157%) and baseline curve (plotted to 40y; Figure 2). Four
extensions of the zero curve beyond the last plotted node; also the fit of
the spline to the digitised yields inside the plotted range."""
import numpy as np
import jax
import jax.numpy as jnp
import common
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

N = 2**15
figs = markets.paper_figures()
out = {}
for which, fig, T_end, sig in (("market", "fig10", 30.0, 0.01157), ("baseline", "fig2", 40.0, 0.01)):
    ref = figs[fig]
    rt = np.asarray(ref["T"])
    m = Model(kappa=0.03, delta=5.0, sigma=sig)
    t = time_grid(T_end)
    mats = jnp.array([T_end - 10, T_end - 5, T_end])
    print(f"\n{which} curve ({fig}); paper at {[float(x) for x in mats]}y:")
    for key in ("Y_eh", "theta", "gamma") if which == "market" else ("Y_eh",):
        print(f"   paper {key:6s}", [ref[key][int(np.argmin(abs(rt - T)))] for T in np.asarray(mats)])
    for ext in ("quadratic", "linear", "flat", "flat_fwd"):
        c = common.paper_curve_ext(which, ext)
        inside = np.abs(BP * np.asarray(c.zero(jnp.asarray(rt[3:]))) - np.asarray(ref["Y"], dtype=float)[3:])
        ybar = volscale.ybar_zeroth(c, 5.0, t)
        d = decompose(simulate(m, c, t, ybar, jax.random.PRNGKey(1), n_paths=N), m, c, mats)
        row = {k: (BP * np.asarray(getattr(d, k))).tolist() for k in ("y_eh", "theta", "gamma")}
        row["fwd_end"] = BP * float(c.forward(T_end))
        row["fwd_end_plus5"] = BP * float(c.forward(T_end + 5))
        row["max_fit_err_inside"] = float(np.nanmax(inside))
        out[f"{which}|{ext}"] = row
        print(f"  {ext:9s} f({T_end:.0f}) {row['fwd_end']:6.1f} f({T_end+5:.0f}) {row['fwd_end_plus5']:6.1f} | "
              f"Y_eh {np.round(row['y_eh'],1)} theta {np.round(row['theta'],1)} gamma {np.round(row['gamma'],1)}"
              f" | spline vs digitised Y inside range: max {row['max_fit_err_inside']:.2f}bp")
save("a8_extrapolation", out)

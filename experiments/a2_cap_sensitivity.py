"""A2: how much does the 10x cap of Remark 8 bind at b > 0? gamma(40y) for cap
levels 5x .. 50x, monthly bootstrap grid, 12 steps a year, 262,144 paths
(common random numbers across cap levels)."""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
figs = markets.paper_figures()
t = time_grid(40.0)
out = {}
for which, fig, b in (("baseline", "fig3", 0.25), ("baseline", "fig3", 0.5), ("baseline", "fig3", 1.0),
                      ("steep", "fig6", 1.0)):
    curve = markets.paper_curve(which)
    paper = figs[fig][f"b={b:.2f}"][-1]
    row = {}
    for cap in (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 50.0):
        m = BASE._replace(b=b, vol_cap=cap)
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)
        prof = simulate(m, curve, t, ybar, jax.random.PRNGKey(11), n_paths=2**18)
        d = decompose(prof, m, curve, jnp.array([40.0]))
        row[cap] = (BP * float(d.gamma[0]), BP * float(d.gamma_se[0]))
    out[f"{which} b={b}"] = {"paper": paper, **{str(k): v for k, v in row.items()}}
    print(f"{which:8s} b={b:4.2f} (paper {paper:6.1f}): " + "  ".join(
        f"{c:g}x {v[0]:6.1f}" for c, v in row.items()) + f"   (s.e. {row[10.0][1]:.1f})", flush=True)
save("a2_cap_sensitivity", out)

"""A2: time-step convergence of gamma at high skew, monthly bootstrap grid,
262,144 paths, two seeds (independent draws at each step size)."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
figs = markets.paper_figures()
out = {}
for which, fig, b in (("baseline", "fig3", 0.5), ("baseline", "fig3", 0.75), ("baseline", "fig3", 1.0),
                      ("steep", "fig6", 1.0)):
    curve = markets.paper_curve(which)
    m = BASE._replace(b=b)
    paper = figs[fig][f"b={b:.2f}"][-1]
    row = {}
    for spy in (12, 24, 48):
        t = time_grid(40.0, steps_per_year=spy)
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)
        g = [decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(s), n_paths=2**18), m, curve,
                       jnp.array([40.0])) for s in (11, 12)]
        vals = [BP * float(d.gamma[0]) for d in g]
        ses = [BP * float(d.gamma_se[0]) for d in g]
        mean, se = float(np.mean(vals)), float(np.sqrt(np.sum(np.square(ses))) / 2)
        row[spy] = (mean, se)
    out[f"{which} b={b}"] = {"paper": paper, **{str(k): v for k, v in row.items()}}
    print(f"{which:8s} b={b:4.2f} (paper {paper:6.1f}): " + "  ".join(
        f"{spy}/yr {v[0]:6.1f} +- {v[1]:.1f} ({v[0]-paper:+.1f})" for spy, v in row.items()), flush=True)
save("a2_step_convergence", out)

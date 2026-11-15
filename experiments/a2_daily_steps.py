"""A2/A3: daily time steps (365 a year) against the step sizes used so far.

gamma(40y) on the A2 cases, both volatility schemes, 1,048,576 paths, two
seeds, monthly bootstrap grid; and the 20y x 5y yield option at the step
sizes of the pricing runs (24/yr) and daily. Nothing is fitted: this asks
whether anything is left of the time-step error at all.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python a2_daily_steps.py
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from common import BP, implied, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale
from slidingcurve import pricing as pr

jax.config.update("jax_enable_x64", True)
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
N = 2**20
figs = markets.paper_figures()
out = {"gamma": {}, "smiles": {}}
t0 = time.time()

print("gamma(40y), bp: mean of seeds 11, 12 (s.e.)")
for which, fig, b in (("baseline", "fig3", 0.0), ("baseline", "fig3", 0.5), ("baseline", "fig3", 1.0),
                      ("steep", "fig6", 1.0)):
    curve = markets.paper_curve(which)
    m = BASE._replace(b=b)
    paper = figs[fig][f"b={b:.2f}"][-1]
    row = {}
    for spy in (12, 48, 365):
        t = time_grid(40.0, steps_per_year=spy)
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)
        for scheme in (("euler",) if b == 0.0 else ("euler", "pc")):
            mm = m._replace(vol_scheme=scheme)
            ds = [decompose(simulate(mm, curve, t, ybar, jax.random.PRNGKey(s), n_paths=N), mm, curve,
                            jnp.array([40.0])) for s in (11, 12)]
            g = float(np.mean([BP * float(d.gamma[0]) for d in ds]))
            se = float(np.sqrt(sum((BP * float(d.gamma_se[0]))**2 for d in ds)) / 2)
            res = float(np.mean([BP * float(d.residual[0]) for d in ds]))
            row[f"{spy}/{scheme}"] = (g, se, res)
    out["gamma"][f"{which} b={b}"] = dict(paper=paper, **row)
    print(f"{which:8s} b={b:4.2f}  paper {paper:6.1f}   [{time.time() - t0:.0f}s]")
    for k, (g, se, res) in row.items():
        print(f"   {k:10s} {g:7.2f} +- {se:.2f}   (paper - model {paper - g:+5.2f}; residual {res:+.3f})",
              flush=True)

print("\n20y x 5y yield option, implied normal vol (bp), offsets from the mean, 262,144 paths")
curve = markets.paper_curve("baseline")
offs = np.array([-200, -120, -40, 0, 40, 120, 200]) * 1e-4
for label, m in (("b=1 (Fig 4)", BASE._replace(b=1.0)), ("eta=1.5 (Fig 8)", BASE._replace(eta=1.5))):
    fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
    res = {}
    for spy in (24, 365):
        _, px, se = pr.yield_option_prices(m, curve, 20.0, offs, fn, jax.random.PRNGKey(7),
                                           n_paths=2**18, steps_per_year=spy)
        res[spy] = [BP * implied(float(p), 0.0, float(o), 20.0) for p, o in zip(px, offs)]
    out["smiles"][label] = res
    print(f"{label}: offsets {[int(o) for o in offs * BP]}   [{time.time() - t0:.0f}s]")
    for spy, v in res.items():
        print(f"   {spy:3d}/yr " + " ".join(f"{x:6.2f}" for x in v))
    print("   daily - 24/yr " + " ".join(f"{a - b_:+6.2f}" for a, b_ in zip(res[365], res[24])), flush=True)
save("a2_daily_steps", out)

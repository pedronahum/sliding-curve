"""A2: does freezing sigma_r over a step matter? Euler against the
predictor-corrector volatility (Model.vol_scheme = "pc"), same normals.

Part 1, convexity: gamma(40y) on the cases of a2_step_convergence, at 12, 4
and 2 steps a year, 262,144 paths, two seeds, monthly bootstrap grid. The
time-step error of each scheme is read against the 48-steps-a-year Euler
value on the same seeds.
Part 2, smiles: the 20y x 5y yield option (library reading), 131,072 paths,
at 24 and 4 steps a year: b = 1 (Figure 4's 20y panel) and b = 0,
eta = 1.5 (Figure 8).
Nothing is fitted.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python a2_predictor_corrector.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from common import BP, implied, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale
from slidingcurve import pricing as pr

jax.config.update("jax_enable_x64", True)
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
figs = markets.paper_figures()
out = {"gamma": {}, "smiles": {}}

print("Part 1: gamma(40y), bp (mean of seeds 11, 12; s.e.)")
for which, fig, b in (("baseline", "fig3", 0.5), ("baseline", "fig3", 1.0), ("steep", "fig6", 1.0)):
    curve = markets.paper_curve(which)
    m = BASE._replace(b=b)
    paper = figs[fig][f"b={b:.2f}"][-1]
    row = {}
    for spy, scheme in ((48, "euler"), (12, "euler"), (12, "pc"), (4, "euler"), (4, "pc"), (2, "euler"), (2, "pc")):
        t = time_grid(40.0, steps_per_year=spy)
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)
        mm = m._replace(vol_scheme=scheme)
        ds = [decompose(simulate(mm, curve, t, ybar, jax.random.PRNGKey(s), n_paths=2**18), mm, curve,
                        jnp.array([40.0])) for s in (11, 12)]
        g = float(np.mean([BP * float(d.gamma[0]) for d in ds]))
        se = float(np.sqrt(sum((BP * float(d.gamma_se[0]))**2 for d in ds)) / 2)
        row[f"{spy}/{scheme}"] = (g, se)
    out["gamma"][f"{which} b={b}"] = dict(paper=paper, **row)
    ref = row["48/euler"][0]
    print(f"{which:8s} b={b:4.2f} paper {paper:6.1f}  48/yr Euler {ref:6.1f} +- {row['48/euler'][1]:.1f}")
    for spy in (12, 4, 2):
        e, p = row[f"{spy}/euler"][0], row[f"{spy}/pc"][0]
        print(f"   {spy:2d}/yr: Euler {e:6.1f} ({e - ref:+5.1f})   PC {p:6.1f} ({p - ref:+5.1f})", flush=True)

print("\nPart 2: 20y x 5y smile, implied normal vol (bp), offsets from the mean")
curve = markets.paper_curve("baseline")
offs = np.array([-200, -120, -40, 0, 40, 120, 200]) * 1e-4
grid = np.arange(1.0, 21.0)
for label, m in (("b=1 (Fig 4)", BASE._replace(b=1.0)), ("eta=1.5 (Fig 8)", BASE._replace(eta=1.5))):
    fn = volscale.bootstrap_fn(curve, m, grid)
    res = {}
    for spy in (24, 4):
        for scheme in ("euler", "pc"):
            mm = m._replace(vol_scheme=scheme)
            mean, px, se = pr.yield_option_prices(mm, curve, 20.0, offs, fn, jax.random.PRNGKey(7),
                                                  n_paths=2**17, steps_per_year=spy)
            vols = [BP * implied(float(p), 0.0, float(o), 20.0) for p, o in zip(px, offs)]
            res[f"{spy}/{scheme}"] = vols
    out["smiles"][label] = res
    print(f"{label}: offsets {list((offs * BP).astype(int))}")
    for k, v in res.items():
        print(f"   {k:9s} " + " ".join(f"{x:6.2f}" for x in v))
    for spy in (24, 4):
        d = np.array(res[f"{spy}/pc"]) - np.array(res[f"{spy}/euler"])
        print(f"   PC - Euler at {spy}/yr: " + " ".join(f"{x:+6.2f}" for x in d), flush=True)
save("a2_predictor_corrector", out)

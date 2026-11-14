"""A2: convergence of gamma in the time step, for the annual (library) and the
monthly bootstrap grid, at high skew and on what already matches. Same seed
throughout (common random numbers across grids at a given step)."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

figs = markets.paper_figures()
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
TARGETS = [("baseline", "fig3", 1.0, 40), ("steep", "fig6", 1.0, 40), ("steep", "fig6", 0.75, 40),
           ("baseline", "fig3", 0.25, 40), ("steep", "fig6", 0.25, 40), ("market", "fig12", 0.367, 30)]


def ref(fig, series, T):
    t = np.asarray(figs[fig]["T"]); return figs[fig][series][int(np.argmin(np.abs(t - T)))]


out = {}
for spy in (12, 24, 48, 96):
    for gridname in ("annual", "monthly"):
        t0 = time.time()
        row = {}
        for which, fig, b, T in TARGETS:
            curve = markets.paper_curve(which)
            m = BASE._replace(b=b, sigma=0.011999 if which == "market" else 0.01)
            H = 30.0 if which == "market" else 40.0
            grid = np.arange(1.0, H + 1) if gridname == "annual" else np.arange(1, int(12 * H) + 1) / 12.0
            t = time_grid(H, steps_per_year=spy)
            ybar = volscale.bootstrap_fn(curve, m, grid)(t)
            d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**16), m, curve,
                          jnp.array([float(T)]))
            paper = ref(fig, "gamma" if fig == "fig12" else f"b={b:.2f}", T)
            row[f"{which} b={b}"] = (BP * float(d.gamma[0]), BP * float(d.gamma_se[0]), paper)
        out[f"{spy}|{gridname}"] = row
        print(f"{spy:3d} steps/yr, {gridname:7s} grid: " + " | ".join(
            f"{k}: {v[0]:6.1f}+-{v[1]:.1f} ({v[0]-v[2]:+5.1f})" for k, v in row.items()) + f" [{time.time()-t0:.0f}s]",
            flush=True)
save("a2_convergence", out)

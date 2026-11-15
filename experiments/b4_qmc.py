"""B4: randomised quasi-Monte Carlo (scrambled Sobol + Brownian bridge)
against the library's pseudo-random antithetic Monte Carlo.

For each quantity, R = 16 independent replications at N paths each (seeds
for MC, scrambles for QMC), N in {2^10, 2^12, 2^14}. The spread across
replications is the standard error of one N-path run; its square ratio is
the variance reduction at equal paths. Plain Sobol (coordinates in time
order, no bridge) shows what the bridge contributes. The pooled means
(262,144 paths) are compared with each other and with the paper: QMC
reduces noise, it does not move a bias.

Quantities (those behind the open items):
  gamma(40y), baseline, b = 0.5 and b = 1        (Figure 3, A2), 12 steps/yr
  20y x 5y yield option ATM vol, b = 0, eta = 1.5 (Figure 8, A3), 24 steps/yr
  same option, b = 1, vol at +200bp from the mean (Figure 4, A1)

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python b4_qmc.py
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from common import BP, implied, save
from slidingcurve import Model, decompose, markets, qmc, simulate, time_grid, volscale
from slidingcurve import pricing as pr

jax.config.update("jax_enable_x64", True)
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
R = 16
SIZES = (2**10, 2**12, 2**14)
METHODS = ("mc", "qmc-bridge", "qmc-plain")
curve = markets.paper_curve("baseline")
figs = markets.paper_figures()


def gamma_fn(b):
    m = BASE._replace(b=b)
    t = time_grid(40.0, 12)
    ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)

    def run(method, n, r):
        if method == "mc":
            p = simulate(m, curve, t, ybar, jax.random.PRNGKey(1000 + r), n_paths=n, n_batches=1)
        else:
            z = qmc.sobol_normals(t.shape[0], 1, n, seed=5000 + r, bridge=method == "qmc-bridge")
            p = simulate(m, curve, t, ybar, None, n_paths=n, n_batches=1, normals=z)
        return BP * float(decompose(p, m, curve, jnp.array([40.0])).gamma[0])
    return run


def option_fn(m, offset):
    fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
    T, spy = 20.0, 24
    n_steps = int(round(T * spy))
    off = np.array([offset])

    def run(method, n, r):
        if method == "mc":
            _, px, _ = pr.yield_option_prices(m, curve, T, off, fn, jax.random.PRNGKey(1000 + r),
                                              n_paths=n, steps_per_year=spy)
        else:
            z = qmc.sobol_normals(n_steps, 1, n, seed=5000 + r, bridge=method == "qmc-bridge")[:, :, 0, :]
            _, px, _ = pr.yield_option_prices(m, curve, T, off, fn, None, n_paths=n,
                                              steps_per_year=spy, normals=z)
        return BP * implied(float(px[0]), 0.0, offset, T)
    return run


QUANTITIES = [
    ("gamma(40y) b=0.5", gamma_fn(0.5), figs["fig3"]["b=0.50"][-1]),
    ("gamma(40y) b=1", gamma_fn(1.0), figs["fig3"]["b=1.00"][-1]),
    ("ATM vol 20y eta=1.5", option_fn(BASE._replace(eta=1.5), 0.0), figs["fig8"]["eta=1.5"][5]),
    ("vol +200 20y b=1", option_fn(BASE._replace(b=1.0), 0.02), figs["fig4"]["20"]["b=1.00"][-1]),
]

out = {}
t0 = time.time()
for name, run, paper in QUANTITIES:
    res = {meth: {n: [run(meth, n, r) for r in range(R)] for n in SIZES} for meth in METHODS}
    sd = {meth: {n: float(np.std(res[meth][n], ddof=1)) for n in SIZES} for meth in METHODS}
    pooled = {meth: (float(np.mean(res[meth][SIZES[-1]])), sd[meth][SIZES[-1]] / np.sqrt(R)) for meth in METHODS}
    out[name] = dict(paper=paper, values=res, sd=sd, pooled=pooled)
    print(f"\n{name} (paper {paper}); s.e. of one run (bp), and variance gain over MC "
          f"[{time.time() - t0:.0f}s]", flush=True)
    for n in SIZES:
        g = lambda meth: (sd["mc"][n] / sd[meth][n]) ** 2
        print(f"  N = {n:6d}: MC {sd['mc'][n]:6.3f}   QMC+bridge {sd['qmc-bridge'][n]:6.3f} (x{g('qmc-bridge'):5.1f})"
              f"   QMC plain {sd['qmc-plain'][n]:6.3f} (x{g('qmc-plain'):5.1f})", flush=True)
    print("  pooled, 16 x 16,384 paths: " + "   ".join(
        f"{meth} {v[0]:7.2f} +- {v[1]:.2f}" for meth, v in pooled.items()), flush=True)
save("b4_qmc", out)

"""A2: is the bootstrap (81) converged in its expiry grid? (81) is the
rectangle-rule discretisation of the continuous condition (80); refine the
grid T_m and watch gamma (common random numbers: same key, same time grid,
only Ybar changes). 24 steps a year."""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale

BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
GRIDS = {"annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12, "fortnightly": 26, "weekly": 52}
out = {}
for which, b, H, sig in (("steep", 1.0, 40.0, 0.01), ("steep", 0.75, 40.0, 0.01), ("baseline", 1.0, 40.0, 0.01),
                         ("baseline", 0.25, 40.0, 0.01), ("market", 0.367, 30.0, 0.011999)):
    curve = markets.paper_curve(which)
    m = BASE._replace(b=b, sigma=sig)
    t = time_grid(H, steps_per_year=24)
    row = {}
    for name, f in GRIDS.items():
        grid = np.arange(1, int(f * H) + 1) / f
        ybar = volscale.bootstrap_fn(curve, m, grid)(t)
        d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**16), m, curve, jnp.array([H]))
        row[name] = BP * float(d.gamma[0])
    out[f"{which} b={b}"] = row
    a = row["annual"]
    print(f"{which:8s} b={b:5.3f}: " + "  ".join(f"{k} {v:6.1f} ({v - a:+.2f})" for k, v in row.items()), flush=True)
save("a2_grid_convergence", out)

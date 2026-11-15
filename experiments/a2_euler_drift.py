"""A2: does a plain Euler P-drift (the literal (52)) move gamma at high skew?

The library advances x with the increment that keeps the benchmark yield an
exact discrete-time martingale (model.py docstring). A direct implementation
of (52) would instead use the left-point Euler drift

    x_new = x + (y - kappa x - mu^Y / A(delta)) dt + sigma_r sqrt(dt) xi,
    mu^Y  = rho + sigma_r^2 B(delta)^2 / (2 delta),  rho from (27) and (43),

which is the same scheme in the limit dt -> 0 but leaves an O(dt) drift in
E^P[Y_1]. At b > 0 the level of Y_1 sets the local vol, so the bias can reach
gamma. Both schemes use the same normals (paired comparison), at fixed step
sizes of 12 and 4 a year; nothing is fitted. Cases as in a2_step_convergence.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python a2_euler_drift.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from common import BP, save
from slidingcurve import Model, decompose, markets, simulate, time_grid, volscale
from slidingcurve.gaussian import A, B
from slidingcurve.model import Profiles, local_vol, variance_step

jax.config.update("jax_enable_x64", True)
BASE = Model(kappa=0.03, delta=5.0, sigma=0.01)
N, NB = 2**18, 16


def simulate_euler(model, curve, t, ybar, key, n_paths=N, n_batches=NB):
    """model.simulate (defaults) with the literal Euler drift of (52)."""
    kappa, delta = model.kappa, model.delta
    dt = t[1] - t[0]
    Bd, Ad, ekd = B(kappa, delta), A(kappa, delta), jnp.exp(-kappa * delta)
    c_y = Bd**2 / (2.0 * delta)
    decay_y, gain_y = jnp.exp(-2.0 * kappa * dt), B(2.0 * kappa, dt)
    per_batch = n_paths // n_batches
    draw_shape = (2, n_batches, per_batch // 2)
    xs = (jnp.arange(t.shape[0]), curve.forward(t), curve.forward(t + delta),
          curve.fwd_yield(t, t + delta), jnp.asarray(ybar))

    def step(state, inp):
        x, y, z = state
        i, f0, f0d, y0, yb = inp
        y1 = y0 + Ad * x + c_y * y
        vol = local_vol(model, y1, yb, z)
        var = vol**2
        rho = (f0d + ekd * (x + y * Bd) - f0 - x) / delta
        mu_y = rho + var * c_y
        stats = jnp.stack([jnp.mean(a, axis=1) for a in (x, mu_y, var, y1, z)])
        xi = jax.random.normal(jax.random.fold_in(key, i), draw_shape)
        xi = jnp.concatenate([xi, -xi], axis=2)
        x_new = x + (y - kappa * x - mu_y / Ad) * dt + vol * jnp.sqrt(dt) * xi[0]
        y_new = y * decay_y + var * gain_y
        return (x_new, y_new, variance_step(model, z, xi[1], dt)), stats

    shape = (n_batches, 2 * draw_shape[2])
    init = (jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape))
    _, stats = jax.lax.scan(step, init, xs)
    s = jnp.transpose(stats, (1, 2, 0))
    return Profiles(t=t, x=s[0], mu_y=s[1], var=s[2], y1=s[3], z=s[4])


figs = markets.paper_figures()
out = {}
print("gamma(40y), bp: library (exact martingale) vs Euler drift, same normals; "
      "and E[Y1(40)] - Y1(0)")
for which, fig, b in (("baseline", "fig3", 0.0), ("baseline", "fig3", 0.5), ("baseline", "fig3", 1.0),
                      ("steep", "fig6", 0.5), ("steep", "fig6", 1.0)):
    curve = markets.paper_curve(which)
    m = BASE._replace(b=b)
    paper = figs[fig][f"b={b:.2f}"][-1]
    y10 = BP * float(curve.fwd_yield(0.0, 5.0))
    for spy in (12, 4):
        t = time_grid(40.0, steps_per_year=spy)
        ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(40.0))(t)
        res = {}
        for name, fn in (("library", lambda k: simulate(m, curve, t, ybar, k, n_paths=N, n_batches=NB)),
                         ("euler", lambda k: simulate_euler(m, curve, t, ybar, k))):
            g, d1, yeh = [], [], []
            for s in (11, 12):
                p = fn(jax.random.PRNGKey(s))
                d = decompose(p, m, curve, jnp.array([40.0]))
                g.append(BP * float(d.gamma[0])); yeh.append(BP * float(d.y_eh[0]))
                d1.append(BP * float(jnp.mean(p.y1[:, -1])) - y10)
            res[name] = dict(gamma=float(np.mean(g)), y_eh=float(np.mean(yeh)), ey1_drift=float(np.mean(d1)))
        dg = res["euler"]["gamma"] - res["library"]["gamma"]
        out[f"{which} b={b} {spy}/yr"] = dict(paper=paper, **res, d_gamma=dg)
        print(f"{which:8s} b={b:4.2f} {spy:2d}/yr  paper {paper:6.1f}  library {res['library']['gamma']:6.1f}"
              f"  euler {res['euler']['gamma']:6.1f}  (diff {dg:+5.2f})   "
              f"E[Y1]-Y1(0): {res['library']['ey1_drift']:+6.2f} / {res['euler']['ey1_drift']:+6.2f}   "
              f"Y_eh: {res['library']['y_eh']:6.1f} / {res['euler']['y_eh']:6.1f}", flush=True)
save("a2_euler_drift", out)

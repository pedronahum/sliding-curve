"""A6: how much of the gap between this code's skew fit and the paper's is
Monte-Carlo noise? Repeat the ATM-pinned fit (mean-referenced strikes) over
pricing seeds and path counts; report the spread of sigma, b and gamma(30y)."""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import Model, calibration, decompose, markets, simulate, time_grid, volscale

curve = markets.paper_curve("market")
offsets, market, _ = markets.paper_smile()
base = Model(kappa=0.03, delta=5.0, sigma=0.01)
t = time_grid(30.0)
out = {}
for n in (2**15, 2**16):
    rows = []
    for s in range(6):
        f = calibration.calibrate_yield_smile(base, curve, 20.0, offsets, market, free=("sigma", "b"),
                                              weights=[1, 1, 30, 1, 1], key=jax.random.PRNGKey(100 + s), n_paths=n)
        m = f.model
        d = decompose(simulate(m, curve, t, volscale.bootstrap_fn(curve, m, np.arange(1.0, 31.0))(t),
                               jax.random.PRNGKey(1), n_paths=2**16), m, curve, jnp.array([30.0]))
        rows.append((m.sigma, m.b, BP * float(d.gamma[0])))
        print(f"  n {n} seed {s}: sigma {100*m.sigma:.4f}% b {m.b:.4f} gamma(30y) {BP*float(d.gamma[0]):.1f}", flush=True)
    r = np.array(rows)
    out[n] = {"rows": rows, "mean": r.mean(0).tolist(), "sd": r.std(0, ddof=1).tolist()}
    print(f"n = {n}: sigma {100*r[:,0].mean():.4f} +- {100*r[:,0].std(ddof=1):.4f}%  b {r[:,1].mean():.4f} +- "
          f"{r[:,1].std(ddof=1):.4f}  gamma(30y) {r[:,2].mean():.1f} +- {r[:,2].std(ddof=1):.1f}bp "
          "(paper 1.1999%, 0.367, 140.3bp)", flush=True)
save("a6_fit_noise", out)

"""B4: variance reduction with the Gaussian-twin control variate.

1. sigma for an 82bp ATM vol (market curve, b = 0): spread over 8 seeds at
   32,768 and 65,536 paths, without and with the control variate
   (README: 1.161% at 32,768 paths, 1.157% at 65,536).
2. The ATM-pinned skew fit (sigma, b) and gamma(30y) over 6 seeds at 32,768
   paths with the control variate (A6 measured the plain spread).
3. Standard-error ratio (plain / control variate) of the 20y yield option
   at three strikes across skew and vol-of-vol.
"""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import (Model, bachelier as bl, calibration, decompose, markets, pricing as pr,
                          simulate, time_grid, volscale)

curve = markets.paper_curve("market")
offsets, market, _ = markets.paper_smile()
base = Model(kappa=0.03, delta=5.0, sigma=0.01)
out = {"atm": {}, "skew": {}, "se_ratio": {}}

print("1. sigma for 82bp ATM, b = 0 (exact: 1.15721%)")
for n in (2**15, 2**16):
    for cv in (False, True):
        s = [calibration.calibrate_yield_smile(base, curve, 20.0, offsets[2:3], [0.0082], free=("sigma",),
                                               key=jax.random.PRNGKey(10 + i), n_paths=n,
                                               control_variate=cv).model.sigma for i in range(8)]
        out["atm"][f"{n}|{cv}"] = s
        print(f"   {n:6d} paths, control variate {str(cv):5s}: sigma {100*np.mean(s):.5f}% +- {100*np.std(s, ddof=1):.5f}%"
              f"  (range {100*min(s):.4f}-{100*max(s):.4f}%)", flush=True)

print("2. skew fit, 32,768 paths, control variate")
t = time_grid(30.0)
rows = []
for i in range(6):
    f = calibration.calibrate_yield_smile(base, curve, 20.0, offsets, market, free=("sigma", "b"),
                                          weights=[1, 1, 30, 1, 1], key=jax.random.PRNGKey(100 + i), n_paths=2**15,
                                          control_variate=True)
    m = f.model
    d = decompose(simulate(m, curve, t, volscale.bootstrap_fn(curve, m, np.arange(1.0, 31.0))(t),
                           jax.random.PRNGKey(1), n_paths=2**16), m, curve, jnp.array([30.0]))
    rows.append((m.sigma, m.b, BP * float(d.gamma[0])))
    print(f"   seed {i}: sigma {100*m.sigma:.4f}% b {m.b:.4f} gamma(30y) {rows[-1][2]:.1f}", flush=True)
r = np.array(rows)
out["skew"] = {"rows": rows, "mean": r.mean(0).tolist(), "sd": r.std(0, ddof=1).tolist()}
print(f"   sigma {100*r[:,0].mean():.4f} +- {100*r[:,0].std(ddof=1):.4f}%  b {r[:,1].mean():.4f} +- {r[:,1].std(ddof=1):.4f}"
      f"  gamma(30y) {r[:,2].mean():.1f} +- {r[:,2].std(ddof=1):.1f}  (plain, A6: +- 0.0095%, 0.0060, 2.7)")

print("3. s.e. ratio plain / control variate, 20y option at mean -200, 0, +200bp")
off = np.array([-0.02, 0.0, 0.02])
for b in (0.0, 0.367, 1.0):
    for eta in (0.0, 0.5):
        m = Model(0.03, 5.0, 0.012, b=b, eta=eta)
        fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
        se = [np.asarray(pr.yield_option_prices(m, curve, 20.0, off, fn, jax.random.PRNGKey(0), 2**15,
                                                control_variate=cv)[2]) for cv in (False, True)]
        ratio = se[0] / se[1]
        out["se_ratio"][f"{b}|{eta}"] = ratio.tolist()
        print(f"   b {b:5.3f} eta {eta}: {np.round(ratio, 1)}  (variance reduction {np.round(ratio**2, 0)})", flush=True)
save("b4_control_variate", out)

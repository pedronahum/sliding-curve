"""B2: where a calibration spends its time. The (sigma, b) yield-smile fit of
Section 9.4 (32,768 paths, 24 steps a year, 20y expiry), instrumented:
compile versus run time of the residual and of its Jacobian, and the
number of evaluations least_squares makes."""
import time
import numpy as np
import jax
import jax.numpy as jnp
from common import save
from slidingcurve import Model, calibration, markets
import slidingcurve.calibration as cal

curve = markets.paper_curve("market")
offsets, market, _ = markets.paper_smile()
base = Model(kappa=0.03, delta=5.0, sigma=0.01)
counts = {"resid": [], "jac": []}
orig_jit = jax.jit
created = []


def timed_jit(fn, *a, **k):
    j = orig_jit(fn, *a, **k)
    name = "resid" if not created else "jac"          # _solve jits the residual first, then its Jacobian
    created.append(name)

    def wrapper(*args):
        t0 = time.perf_counter()
        r = j(*args)
        jax.block_until_ready(r)
        counts[name].append(time.perf_counter() - t0)
        return r
    return wrapper


import sys
from slidingcurve import pricing as _pr
if len(sys.argv) > 1:                                  # e.g. 16, with XLA_FLAGS set accordingly
    _pr.set_path_devices(int(sys.argv[1]))
cal.jax.jit = timed_jit
t0 = time.perf_counter()
fit = calibration.calibrate_yield_smile(base, curve, 20.0, offsets, market, free=("sigma", "b"),
                                        weights=[1, 1, 30, 1, 1], n_paths=2**15)
total = time.perf_counter() - t0
cal.jax.jit = orig_jit
r, j = np.array(counts["resid"]), np.array(counts["jac"])
out = dict(total=total, n_resid=len(r), n_jac=len(j), resid_first=float(r[0]), resid_rest=float(r[1:].mean()),
           jac_first=float(j[0]), jac_rest=float(j[1:].mean()) if len(j) > 1 else None,
           sigma=fit.model.sigma, b=fit.model.b)
print(f"total {total:.1f}s | residual: {len(r)} calls, first {r[0]:.1f}s (compile + run), then {r[1:].mean():.2f}s each"
      f" | jacobian: {len(j)} calls, first {j[0]:.1f}s, then {(j[1:].mean() if len(j)>1 else float('nan')):.2f}s each"
      f" | other {total - r.sum() - j.sum():.1f}s | sigma {100*fit.model.sigma:.4f}% b {fit.model.b:.4f}")
save("b2_profile" + ("_devices" + sys.argv[1] if len(sys.argv) > 1 else ""), out)

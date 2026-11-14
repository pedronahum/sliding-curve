"""B1: the three-parameter (sigma, b, eta) fit on the Section 9.4 smile with
QE + finite differences (library default) and lognormal + autodiff, same
start (the skew fit, eta = 0.5), 32,768 paths. Time, evaluations, result."""
import time
import numpy as np
import jax
from common import save
from slidingcurve import Model, calibration, markets

curve = markets.paper_curve("market")
offsets, market, _ = markets.paper_smile()
out = {}
for sch in ("qe", "lognormal"):
    m0 = Model(0.03, 5.0, 0.0119, b=0.353, eta=0.5, variance_scheme=sch)
    t0 = time.perf_counter()
    f = calibration.calibrate_yield_smile(m0, curve, 20.0, offsets, market, free=("sigma", "b", "eta"),
                                          key=jax.random.PRNGKey(0), n_paths=2**15)
    dt = time.perf_counter() - t0
    out[sch] = dict(time=dt, sigma=f.model.sigma, b=f.model.b, eta=f.model.eta, errors=f.vol_errors_bp.tolist())
    print(f"{sch:9s}: {dt:6.1f}s  sigma {100*f.model.sigma:.3f}% b {f.model.b:.3f} eta {f.model.eta:.3f}"
          f"  errors {np.round(f.vol_errors_bp, 2)}", flush=True)
save("b1_fit_timing", out)

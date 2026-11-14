"""C6: timings on CPU (one core, or several host devices) and GPU.

    ../.venv/bin/python c6_gpu_timings.py cpu
    XLA_FLAGS=--xla_force_host_platform_device_count=16 ../.venv/bin/python c6_gpu_timings.py cpu16
    XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python c6_gpu_timings.py gpu

The GPU (a GB10) shares system RAM, so JAX's default preallocation of 75%
of device memory can exhaust the host: switch it off as above.

Workloads: a 20y pricing simulation (Q^(T+delta), b = 0.35, 24 steps a
year) at 65,536 and 1,048,576 paths; a 40y physical-measure decomposition
(baseline curve, b = 1, 12 steps a year, 65,536 and 1,048,576 paths); and
the (sigma, b) skew fit of Section 9.4. Times exclude compilation (the
first call), except the fit, which is end to end. Results are printed to
check that every backend computes the same numbers (same random numbers:
float64 throughout)."""
import json
import pathlib
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from slidingcurve import (Model, calibration, decompose, markets, pricing as pr, simulate, time_grid,
                          volscale)

label = sys.argv[1]
if label == "cpu16":
    pr.set_path_devices()
out = {"label": label, "devices": [str(d) for d in jax.devices()]}
print(label, jax.devices()[:2], "...", flush=True)


def timed(f, reps=3):
    jax.block_until_ready(f())
    t0 = time.perf_counter()
    for _ in range(reps):
        r = f()
        jax.block_until_ready(r)
    return (time.perf_counter() - t0) / reps, r


mkt = markets.paper_curve("market")
m = Model(0.03, 5.0, 0.0119, b=0.35)
fn = volscale.bootstrap_fn(mkt, m, volscale.bootstrap_grid(20.0))
for n in (2**16, 2**20):
    f = jax.jit(lambda k: pr.simulate_terminal(m, mkt, 20.0, fn, k, n, 24, 25.0)[0])
    dt, x = timed(lambda: f(jax.random.PRNGKey(0)))
    out[f"pricing_{n}"] = dt
    print(f"pricing simulation, {n} paths: {dt:.3f}s   (mean x {float(jnp.mean(x)):.10f})", flush=True)

base = markets.paper_curve("baseline")
mb = Model(0.03, 5.0, 0.01, b=1.0)
t = time_grid(40.0)
ybar = volscale.bootstrap_fn(base, mb, volscale.bootstrap_grid(40.0))(t)
for n in (2**16, 2**20):
    g = jax.jit(lambda k: simulate(mb, base, t, ybar, k, n_paths=n).var)
    dt, v = timed(lambda: g(jax.random.PRNGKey(1)), reps=2)
    d = decompose(simulate(mb, base, t, ybar, jax.random.PRNGKey(1), n_paths=n), mb, base, jnp.array([40.0]))
    out[f"decomposition_{n}"] = dt
    print(f"decomposition simulation, {n} paths: {dt:.3f}s   (gamma(40y) {1e4*float(d.gamma[0]):.4f}bp)", flush=True)

offsets, market, _ = markets.paper_smile()
t0 = time.perf_counter()
fit = calibration.calibrate_yield_smile(Model(0.03, 5.0, 0.01), mkt, 20.0, offsets, market, free=("sigma", "b"),
                                        weights=[1, 1, 30, 1, 1], n_paths=2**15)
out["fit"] = time.perf_counter() - t0
print(f"skew fit, 32,768 paths: {out['fit']:.1f}s   (sigma {100*fit.model.sigma:.6f}%, b {fit.model.b:.6f})", flush=True)
res = pathlib.Path(__file__).parent / "results"
res.mkdir(exist_ok=True)
(res / f"c6_timings_{label}.json").write_text(json.dumps(out, indent=1))

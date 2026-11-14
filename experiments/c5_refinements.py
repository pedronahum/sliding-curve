"""C5: the equilibrium-rate (5.3) and front-end (5.4) refinements of the P-drift.

Market curve of Section 9.4, Gaussian model with sigma = 1.157%, 65,536 paths,
monthly steps. For each drift specification, the decomposition (12) at 2, 5,
10 and 30 years, with Monte-Carlo standard errors, and the paper's
approximations of Y_eh: (36) for the front end (with w_1 = w and r(0) =
f(0, 0)), and r* + (1 - exp(-aT)) / (aT) (r(0) - r*) for the equilibrium rate
(Section 5.3; the model reverts Y_1, not r, so this is a rough guide only)."""
import json
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
from slidingcurve import Model, decompose, front_end_path, markets, simulate, time_grid, volscale

curve = markets.paper_curve("market")
m = Model(kappa=0.03, delta=5.0, sigma=0.01157)
t = time_grid(30.0)
mats = np.array([2.0, 5.0, 10.0, 30.0])
ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(30.0))(t)
r0 = float(curve.forward(0.0))


def run(**kw):
    p = simulate(m, curve, t, ybar, jax.random.PRNGKey(5), n_paths=2**16, **kw)
    return decompose(p, m, curve, jnp.asarray(mats))


def approx_front(Tc, c, form):
    out = []
    for T in mats:
        u = np.linspace(0.0, T, 2001)
        w = np.exp(-c * u / Tc) if form == "exp" else np.maximum(1 - u / Tc, 0) ** c
        g = (1 - w) * (np.asarray(curve.forward(u)) - r0)
        out.append(float(curve.zero(T)) - np.trapezoid(g, u) / T)
    return np.array(out)


def approx_eq(a, ystar):
    return ystar + (1 - np.exp(-a * mats)) / (a * mats) * (r0 - ystar)


variants = [("martingale (26)", {}, None)]
for Tc, c, form in [(2.5, 1.0, "exp"), (2.5, 3.0, "exp"), (2.5, 1.0, "power")]:
    variants.append((f"front end Tc={Tc} c={c} {form}",
                     dict(ey_path=front_end_path(curve, m.delta, t, Tc, c, form)), approx_front(Tc, c, form)))
for a, ys in [(0.2, 0.035), (0.1, 0.045)]:
    variants.append((f"equilibrium a={a} Y*={100*ys:.1f}%", dict(reversion=(a, ys)), approx_eq(a, ys)))

bp = lambda v: [round(1e4 * float(x), 2) for x in v]
res = {}
print("T =", mats, "; all bp; s.e. in brackets")
for name, kw, apx in variants:
    d = run(**kw)
    row = dict(y=bp(d.y), y_eh=bp(d.y_eh), y_eh_se=bp(d.y_eh_se), theta=bp(d.theta), theta_se=bp(d.theta_se),
               gamma=bp(d.gamma), residual=[round(1e4 * float(x), 4) for x in d.residual],
               y_eh_paper_approx=None if apx is None else bp(apx))
    res[name] = row
    print(f"\n{name}")
    for k in ("y", "y_eh", "theta", "gamma"):
        se = row.get(k + "_se")
        print(f"  {k:6s}", "  ".join(f"{v:8.2f}" + (f" ({s:.2f})" if se else "") for v, s in zip(row[k], se or row[k])))
    if apx is not None:
        print("  Y_eh paper approx", "  ".join(f"{v:8.2f}" for v in row["y_eh_paper_approx"]))
    print("  residual", row["residual"])
out = pathlib.Path(__file__).parent / "results"
(out / "c5_refinements.json").write_text(json.dumps(res, indent=1))

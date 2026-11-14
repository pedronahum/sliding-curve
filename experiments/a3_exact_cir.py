"""A3/B1: is QE biased at eta = 1.5 against exact CIR sampling? 20y x 5y ATM
yield-option vol at sigma = 1%, b = 0 (Figure 8 set-up), 131,072 paths,
four independent seeds per scheme, and the convexity this implies after
recalibrating sigma to 70.86bp (gamma scales as sigma^2 at b = 0 since
E[z] = 1 in every scheme)."""
import numpy as np
import jax
from common import BP, smile, save, exact_cir, scheme_paths, yield_from_z
from slidingcurve import Model, markets, volscale

N = 2**17
curve = markets.paper_curve("baseline")
fn = lambda t: volscale.ybar_zeroth(curve, 5.0, t)
out = {}
for eta in (1.0, 1.5):
    m = Model(0.03, 5.0, 0.01, eta=eta)
    # the lognormal scheme arrives with B1
    for sch in ("exact", "qe") + (("lognormal",) if "variance_scheme" in Model._fields else ()):
        v = []
        for s in range(4):
            z = exact_cir(eta, 0.1, N, 100 + s) if sch == "exact" else scheme_paths(
                m._replace(variance_scheme=sch), N, jax.random.PRNGKey(200 + s))
            y1 = yield_from_z(m, curve, fn, z, jax.random.PRNGKey(300 + s))
            v.append(BP * smile(y1, y1.mean() + np.array([0.0]), y1.mean(), 20.0)[0])
        v = np.array(v)
        out[f"{eta}|{sch}"] = dict(atm=v.tolist(), mean=float(v.mean()), se=float(v.std(ddof=1) / 2))
        # linear recalibration to the exact Gaussian 70.86bp: gamma(40) = 119.0 (sigma / 1%)^2
        print(f"eta {eta}, {sch:9s}: ATM {v.mean():6.2f} +- {v.std(ddof=1)/2:.2f}  (runs {np.round(v,2)})"
              f"  -> gamma(40y) at fixed ATM ~ {119.0*(70.86/v.mean())**2:6.1f}", flush=True)
save("a3_exact_cir", out)

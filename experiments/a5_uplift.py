"""A5: Section 8.2's uplift for eta = 0.6, kappa_z = 0.1 over ten years.

The paper's figures are the second-order approximation (65),
uplift ~ sigma_rel^2 / 4 with sigma_rel^2 = Var(V) / E(V)^2, where
V = int_0^T w(t) z(t) dt and w(t) = exp(-2 kappa (T - t)) is the weight with
which the variance at t reaches the rate at expiry T. This script computes
(1) sigma_rel^2 exactly (double integral of the CIR covariance),
(2) the approximation 1 + sigma_rel^2/4 (the paper's numbers),
(3) the exact Jensen ratio E(V)/E(sqrt V)^2 by exact CIR sampling, and
(4) the uplift the full model gives on recalibrating sigma to an ATM option
    on the benchmark yield and on the simple rate (caplet), b = 0.
"""
import numpy as np
import jax
from common import save
from slidingcurve import Model, bachelier as bl, calibration, markets, pricing as pr, volscale

ETA, KZ, T = 0.6, 0.1, 10.0
PAPER = {0.0: 0.15, 0.03: 0.16, 0.05: 0.175}


def rel_var(kappa, n=2001):
    t = np.linspace(0.0, T, n)
    w = np.exp(-2 * kappa * (T - t))
    s, u = np.meshgrid(t, t, indexing="ij")
    var_min = ETA**2 / (2 * KZ) * (1 - np.exp(-2 * KZ * np.minimum(s, u)))   # z0 = theta = 1
    cov = np.exp(-KZ * np.abs(s - u)) * var_min
    ww = np.outer(w, w)
    ev = np.trapezoid(w, t)
    return np.trapezoid(np.trapezoid(ww * cov, t, axis=1), t) / ev**2


def jensen_exact(kappa, paths=400_000, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps
    c = ETA**2 * (1 - np.exp(-KZ * dt)) / (4 * KZ)
    d = 4 * KZ / ETA**2
    z = np.ones(paths)
    V = np.zeros(paths)
    for i in range(steps):
        t0 = i * dt
        z_new = c * rng.noncentral_chisquare(d, z * np.exp(-KZ * dt) / c)
        V += 0.5 * dt * (np.exp(-2 * kappa * (T - t0)) * z + np.exp(-2 * kappa * (T - t0 - dt)) * z_new)
        z = z_new
    r = V.mean() / np.mean(np.sqrt(V)) ** 2
    # standard error by batches
    b = np.array([vb.mean() / np.mean(np.sqrt(vb)) ** 2 for vb in np.array_split(V, 20)])
    return r, b.std(ddof=1) / np.sqrt(20)


def model_uplift(kappa, instrument):
    curve = markets.paper_curve("baseline")
    base = Model(kappa=kappa, delta=5.0, sigma=0.01, kappa_z=KZ)
    key = jax.random.PRNGKey(0)
    ybar = lambda t: volscale.ybar_zeroth(curve, 5.0, t)
    if instrument == "yield":
        _, p, _ = pr.yield_option_prices(base, curve, T, np.array([0.0]), ybar, key, 2**16)
        v0 = bl.implied_normal_vol(np.asarray(p), 0.0, np.array([0.0]), T)
        fit = calibration.calibrate_yield_smile(base._replace(eta=ETA), curve, T, np.array([0.0]), v0,
                                                free=("sigma",), key=key, n_paths=2**16)
    else:
        from slidingcurve import markets as mk
        swp = pr.Swaption.caplet(T, 5.0)
        fwd, ann = float(pr.forward_swap_rate(curve, swp)), float(pr.annuity(curve, swp))
        k = np.array([fwd])
        p0 = np.asarray(pr.swaption_prices(base, curve, swp, k, ybar, key, n_paths=2**16)[0])
        q = mk.Quotes(swp, fwd, ann, k, p0, bl.implied_normal_vol(p0 / ann, fwd, k, T))
        fit = calibration.calibrate(base._replace(eta=ETA), curve, q, free=("sigma",), ybar_fn=ybar,
                                    key=key, n_paths=2**16)
    return (fit.model.sigma / 0.01) ** 2 - 1


out = {}
print(" kappa | rel. var | 1 + rel.var/4 (paper) | exact E(V)/E(sqrt V)^2 | model, yield option | model, caplet")
for k in (0.0, 0.03, 0.05):
    rv = rel_var(k)
    ex, ex_se = jensen_exact(k)
    my, mc = model_uplift(k, "yield"), model_uplift(k, "caplet")
    out[k] = dict(rel_var=rv, approx=rv / 4, paper=PAPER[k], exact=ex - 1, exact_se=ex_se, model_yield=my,
                  model_caplet=mc)
    print(f" {k:5.2f} | {rv:8.3f} | {100*rv/4:5.1f}% ({100*PAPER[k]:4.1f}%)        | "
          f"{100*(ex-1):5.2f}% +- {100*ex_se:.2f}%        | {100*my:5.2f}%              | {100*mc:5.2f}%", flush=True)
save("a5_uplift", out)

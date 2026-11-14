"""A7: the three-parameter (sigma, b, eta) fit to the Section 9.4 smile.

1. Profile: eta fixed on a grid, (sigma, b) fitted by autodiff (ATM-pinned
   weights, mean-referenced strikes), giving the residual and gamma(30y)
   along the valley; the paper's point (1.492%, 0.417, 0.779) and this
   code's (1.80%, 0.36, 1.39) lie on it if the valley is the explanation.
2. The residuals at the two stated points directly, with Monte-Carlo s.e.
3. The spurious basin: the full fit started from sigma 2.6%, eta 2.
"""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, save
from slidingcurve import (Model, bachelier as bl, calibration, decompose, markets, pricing as pr,
                          simulate, time_grid, volscale)

N = 2**15
curve = markets.paper_curve("market")
offsets, market, _ = markets.paper_smile()
base = Model(kappa=0.03, delta=5.0, sigma=0.012, b=0.36)
W = [1, 1, 30, 1, 1]
key = jax.random.PRNGKey(0)
t30 = time_grid(30.0)


def gamma30(m):
    d = decompose(simulate(m, curve, t30, volscale.bootstrap_fn(curve, m, np.arange(1.0, 31.0))(t30),
                           jax.random.PRNGKey(1), n_paths=2**15), m, curve, jnp.array([30.0]))
    return BP * float(d.gamma[0]), BP * float(d.gamma_se[0])


def errors(m):
    mean, p, se = pr.yield_option_prices(m, curve, 20.0, offsets, volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0)),
                                         key, N)
    v = bl.implied_normal_vol(np.asarray(p), 0.0, offsets, 20.0)
    return BP * (v - market), BP * np.asarray(se / bl.normal_vega(0.0, offsets, v, 20.0))


out = {"profile": {}, "points": {}}
print("1. profile in eta (sigma, b fitted, ATM pinned)")
for eta in (0.0, 0.4, 0.6, 0.779, 1.0, 1.2, 1.39, 1.6):
    f = calibration.calibrate_yield_smile(base._replace(eta=eta), curve, 20.0, offsets, market,
                                          free=("sigma", "b"), weights=W, key=key, n_paths=N)
    rms = float(np.sqrt(np.mean(f.vol_errors_bp ** 2)))
    g, gse = gamma30(f.model)
    out["profile"][eta] = dict(sigma=f.model.sigma, b=f.model.b, rms=rms, errors=f.vol_errors_bp.tolist(),
                               gamma30=g, gamma30_se=gse)
    print(f"  eta {eta:5.3f}: sigma {100*f.model.sigma:.3f}% b {f.model.b:.3f} | errors {np.round(f.vol_errors_bp,2)} "
          f"rms {rms:.2f}bp | gamma(30y) {g:.1f} +- {gse:.1f}", flush=True)

print("2. the stated points")
for name, m in (("paper", base._replace(sigma=0.01492, b=0.417, eta=0.779)),
                ("this code (README)", base._replace(sigma=0.0180, b=0.36, eta=1.39))):
    e, se = errors(m)
    g, gse = gamma30(m)
    out["points"][name] = dict(errors=e.tolist(), se=se.tolist(), gamma30=g, gamma30_se=gse)
    print(f"  {name:20s}: errors {np.round(e,2)} (s.e. {se.max():.2f}) rms {np.sqrt(np.mean(e**2)):.2f}bp | "
          f"gamma(30y) {g:.1f} +- {gse:.1f}", flush=True)

print("3. full fit from the spurious-basin start")
f = calibration.calibrate_yield_smile(base._replace(sigma=0.026, eta=2.0), curve, 20.0, offsets, market,
                                      free=("sigma", "b", "eta"), key=key, n_paths=N)
out["spurious"] = dict(sigma=f.model.sigma, b=f.model.b, eta=f.model.eta, errors=f.vol_errors_bp.tolist())
print(f"  -> sigma {100*f.model.sigma:.3f}% b {f.model.b:.3f} eta {f.model.eta:.3f} errors {np.round(f.vol_errors_bp,2)}")
save("a7_three_param", out)

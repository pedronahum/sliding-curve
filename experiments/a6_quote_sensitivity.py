"""A6: how much of the gap between our skew fit and the paper's can the
quotes themselves explain?

Our five market quotes are read off Figure 11. Visible markers read to about
0.05bp (25 pixels per bp), but at -50bp and at the money the paper's fitted
square is drawn on top of the market dot, and tools/digitise_figures.py sets
the hidden market value equal to the fit there: the true quote can be up to
a marker radius (~0.4bp) away. Questions, with the quasi-random pricer so
that our own noise is negligible (scrambled Sobol + Brownian bridge):

  1. The fit (sigma, b) to the digitised quotes, ATM pinned (weight 30),
     strikes from the mean, monthly bootstrap grid, 24 steps a year; spread
     over four scrambles.
  2. The sensitivity of sigma, b and gamma(30y) to each quote (linearised
     weighted least squares, Jacobian by central differences on common
     quasi-random numbers).
  3. The smallest change of the quotes (Euclidean, bp) that makes the
     paper's (1.1999%, 0.367) the fit: all quotes free; only the two hidden
     quotes free; ATM held at 82bp. Checked by refitting at the shifted
     quotes.
Nothing is fitted to the paper's answer; step 3 measures a distance.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python a6_quote_sensitivity.py
"""
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares

from common import BP, save
from slidingcurve import (Model, bachelier as bl, decompose, markets, pricing as pr, qmc, simulate,
                          time_grid, volscale)

jax.config.update("jax_enable_x64", True)
P = 2**16
T, SPY = 20.0, 24
curve = markets.paper_curve("market")
offsets, market, paper_fit = markets.paper_smile()
quotes = BP * market
W = np.array([1.0, 1.0, 30.0, 1.0, 1.0])
BASE = Model(kappa=0.03, delta=5.0, sigma=0.012)
PAPER = np.array([1.1999, 0.367])
grid = volscale.bootstrap_grid(20.0)
t0 = time.time()


def normals(seed):
    return qmc.sobol_normals(int(T * SPY), 1, P, seed=seed)[:, :, 0, :]


def vols(theta, z):
    m = BASE._replace(sigma=theta[0] / 100.0, b=theta[1])
    _, px, _ = pr.yield_option_prices(m, curve, T, offsets, volscale.bootstrap_fn(curve, m, grid), None,
                                      n_paths=P, steps_per_year=SPY, normals=z)
    return BP * np.asarray(bl.implied_normal_vol(np.asarray(px), 0.0, offsets, T))


def fit(q, z, x0=(1.19, 0.355)):
    r = least_squares(lambda th: W * (vols(th, z) - q), x0, diff_step=1e-4, x_scale=[0.01, 0.01],
                      xtol=1e-10, ftol=1e-12)
    return r.x


def gamma30(theta, z30):
    m = BASE._replace(sigma=theta[0] / 100.0, b=theta[1])
    t = time_grid(30.0)
    ybar = volscale.bootstrap_fn(curve, m, volscale.bootstrap_grid(30.0))(t)
    p = simulate(m, curve, t, ybar, None, n_paths=P, n_batches=1, normals=z30)
    return BP * float(decompose(p, m, curve, jnp.array([30.0])).gamma[0])


out = {"quotes": quotes.tolist(), "paper_fit_vols": (BP * paper_fit).tolist()}
print("digitised quotes:", np.round(quotes, 2), "  paper's fitted vols:", np.round(BP * paper_fit, 2))

# 1. fit, four scrambles
fits = []
for s in range(4):
    z = normals(100 + s)
    th = fit(quotes, z)
    fits.append(th)
    print(f"  scramble {s}: sigma {th[0]:.4f}%  b {th[1]:.4f}   [{time.time() - t0:.0f}s]", flush=True)
fits = np.array(fits)
th0 = fits.mean(0)
print(f"1. fit: sigma {th0[0]:.4f} +- {fits[:, 0].std(ddof=1):.4f}%  b {th0[1]:.4f} +- {fits[:, 1].std(ddof=1):.4f}"
      f"   (paper {PAPER[0]}%, {PAPER[1]})")
out["fit"] = {"per_scramble": fits.tolist(), "mean": th0.tolist()}

# 2. sensitivities on scramble 0
z = normals(100)
th_fit = fits[0]
h = np.array([0.002, 0.002])
J = np.stack([(vols(th_fit + e, z) - vols(th_fit - e, z)) / (2 * e[i]) for i, e in enumerate(np.diag(h))], 1)
Wd = np.diag(W**2)
M = np.linalg.solve(J.T @ Wd @ J, J.T @ Wd)                          # d theta / d quotes, [2, 5]
z30 = qmc.sobol_normals(361, 1, P, seed=200)
g_fit = gamma30(th_fit, z30)
dg = np.array([(gamma30(th_fit + e, z30) - gamma30(th_fit - e, z30)) / (2 * e[i]) for i, e in enumerate(np.diag(h))])
G = dg @ M                                                              # d gamma30 / d quotes
g_paper = gamma30(PAPER, z30)
print(f"\n2. per +1bp in each quote (-200, -50, ATM, +50, +200):")
print("   sigma (%):    " + " ".join(f"{x:+.4f}" for x in M[0]))
print("   b:            " + " ".join(f"{x:+.4f}" for x in M[1]))
print("   gamma30 (bp): " + " ".join(f"{x:+.2f}" for x in G))
print(f"   gamma(30y) at our fit {g_fit:.2f}, at the paper's parameters {g_paper:.2f} (paper 140.3)"
      f"   [{time.time() - t0:.0f}s]", flush=True)
out["sensitivity"] = {"J": J.tolist(), "dtheta_dq": M.tolist(), "dgamma_dq": G.tolist(),
                      "gamma30_fit": g_fit, "gamma30_paper_params": g_paper}

# 3. smallest quote changes that move the fit to the paper's point
dth = PAPER - th_fit
cases = {"all quotes": [0, 1, 2, 3, 4], "hidden quotes only (-50, ATM)": [1, 2], "ATM held at 82bp": [0, 1, 3, 4]}
out["min_shift"] = {}
print(f"\n3. smallest quote change (bp) that makes ({PAPER[0]}%, {PAPER[1]}) the fit; refit to check")
for name, free in cases.items():
    Mf = M[:, free]
    dq = np.zeros(5)
    dq[free] = Mf.T @ np.linalg.solve(Mf @ Mf.T, dth)
    th_chk = fit(quotes + dq, z, x0=th_fit)
    out["min_shift"][name] = {"dq": dq.tolist(), "norm": float(np.linalg.norm(dq)), "refit": th_chk.tolist()}
    print(f"   {name:30s}: dq = {' '.join(f'{x:+.2f}' for x in dq)}  (|dq| {np.linalg.norm(dq):.2f})"
          f"  -> refit sigma {th_chk[0]:.4f}%, b {th_chk[1]:.4f}   [{time.time() - t0:.0f}s]", flush=True)
save("a6_quote_sensitivity", out)

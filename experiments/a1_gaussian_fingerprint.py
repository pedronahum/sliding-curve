"""A1: can any Gaussian pricing convention produce Figure 4's b = 0 smiles?

At b = 0 the benchmark yield is Gaussian with the same variance in every
measure (the measure changes are deterministic drifts), so any convention is
price(K) = c * Bachelier(m, K, s), inverted with forward F. Fit (c, m - F) to
the paper's b = 0 points and report the residual; a convention explains the
figure only if a natural (c, m - F) gives the shape.
"""
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
from slidingcurve import markets, gaussian as g

BP = 1e4
kappa, delta, sigma = 0.03, 5.0, 0.01
fig4 = markets.paper_figures()["fig4"]


def bach(F, K, v, T, put=False):
    s = v * np.sqrt(T); d = (F - K) / s
    c = (F - K) * norm.cdf(d) + s * norm.pdf(d)
    return c - (F - K) if put else c


def implied(price, F, K, T, put):
    lo = max(K - F, 0) if put else max(F - K, 0)
    if price <= lo + 1e-14:
        return np.nan
    try:
        return brentq(lambda v: bach(F, K, v, T, put) - price, 1e-6, 0.2)
    except ValueError:
        return np.nan


def vols(c, shift, F, K, s, T, mode):
    out = []
    for k in K:
        put = mode == "put" or (mode == "otm" and k < F)
        out.append(implied(c * bach(F + shift, k, s, T, put), F, k, T, put))
    return np.array(out)


for T in (5, 10, 15, 20):
    r = fig4[str(T)]
    F = r["atm"] * 1e-4
    K = F + 1e-4 * np.asarray(r["offsets"])
    paper = np.array(r["b=0.00"], dtype=float)
    s = g.A(kappa, delta) * sigma * np.sqrt(float(g.B(2 * kappa, T)) / T)
    print(f"\n{T}y: Gaussian vol {BP*s:.2f}bp; paper b=0: {np.round(paper,1)}")
    for mode in ("call", "put", "otm"):
        def res(p):
            v = vols(p[0], p[1] * 1e-4, F, K, s, T, mode)
            e = BP * v - paper
            return np.where(np.isnan(e), 0.0, e) + np.where(np.isnan(v) & ~np.isnan(paper), 20.0, 0.0) \
                + np.where(~np.isnan(v) & np.isnan(paper), 5.0, 0.0)
        best = min((least_squares(res, [c0, d0]) for c0 in (0.9, 1.0, 1.1) for d0 in (-30, 0, 30)),
                   key=lambda r_: r_.cost)
        v = BP * vols(best.x[0], best.x[1] * 1e-4, F, K, s, T, mode)
        print(f"  {mode:4s}: c = {best.x[0]:.4f}, mean - F = {best.x[1]:+6.2f}bp, rms {np.sqrt(2*best.cost/len(K)):.2f}bp -> {np.round(v,1)}")

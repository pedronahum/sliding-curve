"""A3/A4/A2: how many Monte-Carlo paths did the paper use? Estimated from
the paper's own noise, then used to test the disputed points.

PRE-REGISTERED PLAN (committed before the first run; see git history)

Hypothesis. The paper's Monte-Carlo figures differ from precise values by
its own sampling error, from a single run of N paths (N not stated in the
paper). If so, one N explains the scatter of every undisputed point, and
the disputed points (A3, A4, A2) are ordinary draws given that N.

Quantities (basis points), the paper's value read off its figure:
  F3, F6  gamma(20y), gamma(40y), b in {0.25, 0.5, 0.75, 1}   (Section 9.2)
  F7      gamma(20y), gamma(40y), eta in {0.5, 1, 1.5}, sigma = 1%
  F9      the same with sigma recalibrated to the eta = 0 ATM vol of the
          same pricing run (as examples/stochvol_sweep.py does)
  F8      20y x 5y smile at curve forward + {-200, 0, +200}bp, eta in
          {0, 0.5, 1, 1.5} (absolute strikes, as in A4)
  F11     the paper's fitted vols at its parameters (1.1999%, 0.367),
          strikes mean + {-200, -50, 0, +50, +200}bp
  F12     gamma(20y), gamma(30y) at the same parameters
Excluded: b = 0 and eta = 0 decompositions (no sampling noise), Figure 4
(A1), Figure 13 (densities), Figures 2, 5, 10 (Gaussian; closed forms).

Noise model. d = paper - precise ~ N(0, S1 / N + S / R + D): S1 is the per-
path covariance of a single run, measured from R replicate runs of this
code with independent seeds (S at n paths; S1 = n S); S / R is the error of
the precise value (the replicate mean); D = (0.3bp)^2 I is digitisation.
Two estimators, reported separately: plain Monte Carlo (independent paths)
and antithetic pairs (this library's default); N counts paths, both members
of a pair included. Three structures for how the paper shared seeds:
  A  one seed for everything (as this repository's example scripts do)
  C  one seed per figure (cross-figure covariance set to zero)
  B  one seed per curve (covariance block-diagonal per curve)

Calibration set: everything except the held-out (disputed) points
  H_A3  F9, eta = 1.5     H_A4  F8, eta = 0     H_A2  F3 and F6, b >= 0.75
N is estimated by maximum likelihood on the calibration set, with a 95%
profile-likelihood interval; goodness of fit is the chi-square of the
whitened calibration residuals; per-figure estimates check consistency.
Each held-out group is then tested conditionally on the calibration
residuals (Gaussian conditioning, which matters under A and C).

Decision rules, fixed in advance:
  supported   calibration fit acceptable (chi-square p > 0.01), per-figure
              intervals mutually overlapping, and the held-out group's
              conditional p > 0.05 at the pooled estimate;
  refuted     per-figure intervals disjoint, or the held-out group's p < 0.01
              at every N inside the pooled interval;
  otherwise   inconclusive.
Caveat stated in advance: any systematic model difference inside the
calibration set inflates the apparent noise, biasing N low and making
held-out points look more plausible than they are. The chi-square and the
per-figure consistency are the guard against that.

    XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python a3_paper_path_count.py [R]
    ../.venv/bin/python a3_paper_path_count.py --analyse      # from the saved replicates
"""
import json
import pathlib
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from scipy import optimize, stats

from slidingcurve import (Model, bachelier as bl, decompose, markets, pricing as pr, simulate, time_grid,
                          volscale)
from slidingcurve.gaussian import A, B

BP = 1e4
N_REP = 2**14                # paths per replicate run (antithetic: 2^13 pairs; plain: first 2^13 paths)
RES = pathlib.Path(__file__).resolve().parent / "results"
NPZ = RES / "a3_paper_path_count.npz"
DIGITISATION = 0.3           # bp

figs = markets.paper_figures()


def paper_at(fig, key, T):
    t = np.asarray(figs[fig]["T"])
    v = np.array(figs[fig][key], dtype=float)
    return float(v[np.argmin(np.abs(t - T))])


# ----------------------------------------------------------------------------- quantities
# Each: (figure, curve label, point label, paper value, held-out group or None)
Q = []
for fig, which in (("fig3", "baseline"), ("fig6", "steep")):
    for b in (0.25, 0.5, 0.75, 1.0):
        for T in (20.0, 40.0):
            Q.append((fig, f"b={b}", f"{T:.0f}y", paper_at(fig, f"b={b:.2f}", T), "A2" if b >= 0.75 else None))
for fig in ("fig7", "fig9"):
    for eta in (0.5, 1.0, 1.5):
        for T in (20.0, 40.0):
            held = "A3" if (fig == "fig9" and eta == 1.5) else None
            Q.append((fig, f"eta={eta}", f"{T:.0f}y", paper_at(fig, f"eta={eta}", T), held))
F8_OFF = (-200.0, 0.0, 200.0)
for eta in (0.0, 0.5, 1.0, 1.5):
    offs = list(figs["fig8"]["offsets"])
    for o in F8_OFF:
        Q.append(("fig8", f"eta={eta}", f"{o:+.0f}", float(figs["fig8"][f"eta={eta}"][offs.index(o)]),
                  "A4" if eta == 0.0 else None))
SM_OFF, _, SM_FIT = markets.paper_smile()
for o, v in zip(SM_OFF, SM_FIT):
    Q.append(("fig11", "fit", f"{BP*o:+.0f}", BP * float(v), None))
for T in (20.0, 30.0):
    Q.append(("fig12", "fit", f"{T:.0f}y", paper_at("fig12", "gamma", T), None))


# ----------------------------------------------------------------------------- simulation
def make_dec(curve, t, mats, antithetic, n):
    mats = jnp.asarray(mats)

    @jax.jit
    def f(sigma, b, eta, ybar, key):
        m = Model(0.03, 5.0, sigma, b=b, eta=eta, kappa_z=0.1)
        p = simulate(m, curve, t, ybar, key, n_paths=n, antithetic=antithetic)
        return BP * decompose(p, m, curve, mats).gamma
    return f


def make_y1(curve, T, ybar_fn):
    fwd = curve.fwd_yield(T, T + 5.0)
    Ad, cy = A(0.03, 5.0), B(0.03, 5.0) ** 2 / 10.0

    @jax.jit
    def f(sigma, b, eta, key):
        m = Model(0.03, 5.0, sigma, b=b, eta=eta, kappa_z=0.1)
        x, y = pr.simulate_terminal(m, curve, T, ybar_fn, key, N_REP, 24, measure_maturity=T + 5.0)[:2]
        return fwd + Ad * x + cy * y
    return f


def vols(y1, strikes, T):
    """Implied normal vols (bp) around the sample mean, from a sample of Y_1(T)."""
    m = float(np.mean(y1))
    p = np.mean(np.maximum(y1[None, :] - np.asarray(strikes)[:, None], 0.0), axis=1)
    return BP * bl.implied_normal_vol(p, m, np.asarray(strikes), T), m


def simulate_replicates(R):
    base, steep, mkt = (markets.paper_curve(w) for w in ("baseline", "steep", "market"))
    t40, t30 = time_grid(40.0, 12), time_grid(30.0, 12)
    m0 = Model(kappa=0.03, delta=5.0, sigma=0.01)
    zeroth = lambda c: (lambda t: volscale.ybar_zeroth(c, 5.0, t))
    ybar = {(w, b): volscale.bootstrap_fn(c, m0._replace(b=b), volscale.bootstrap_grid(40.0))(t40)
            for w, c in (("baseline", base), ("steep", steep)) for b in (0.25, 0.5, 0.75, 1.0)}
    fit = m0._replace(sigma=0.011999, b=0.367)
    ybar12 = volscale.bootstrap_fn(mkt, fit, volscale.bootstrap_grid(30.0))(t30)
    ybar0 = volscale.ybar_zeroth(base, 5.0, t40)
    dec = {(c, a): make_dec(cv, t, mats, a, N_REP if a else N_REP // 2)
           for a in (False, True)
           for c, cv, t, mats in (("baseline", base, t40, [20.0, 40.0]), ("steep", steep, t40, [20.0, 40.0]),
                                  ("market", mkt, t30, [20.0, 30.0]))}
    y1_base = make_y1(base, 20.0, zeroth(base))
    y1_mkt = make_y1(mkt, 20.0, volscale.bootstrap_fn(mkt, fit, volscale.bootstrap_grid(20.0)))
    F20 = float(base.fwd_yield(20.0, 25.0))

    out = {False: [], True: []}
    t0 = time.perf_counter()
    for r in range(R):
        kd, kp = jax.random.fold_in(jax.random.PRNGKey(1001), r), jax.random.fold_in(jax.random.PRNGKey(2002), r)
        y8 = {eta: np.asarray(y1_base(0.01, 0.0, eta, kp)) for eta in (0.0, 0.5, 1.0, 1.5)}
        y11 = np.asarray(y1_mkt(fit.sigma, fit.b, 0.0, kp))
        for anti in (False, True):
            cut = (lambda y: y) if anti else (lambda y: y[: N_REP // 2])
            row = []
            for w in ("baseline", "steep"):
                for b in (0.25, 0.5, 0.75, 1.0):
                    row += list(np.asarray(dec[w, anti](0.01, b, 0.0, ybar[w, b], kd)))
            for eta in (0.5, 1.0, 1.5):                                       # F7
                row += list(np.asarray(dec["baseline", anti](0.01, 0.0, eta, ybar0, kd)))
            y0 = cut(y8[0.0])
            target = vols(y0, [np.mean(y0)], 20.0)[0][0]
            for eta in (0.5, 1.0, 1.5):                                       # F9
                s = 0.01
                for _ in range(5):                      # vol is close to proportional to sigma
                    y = cut(np.asarray(y1_base(s, 0.0, eta, kp)))
                    s *= target / vols(y, [np.mean(y)], 20.0)[0][0]
                row += list(np.asarray(dec["baseline", anti](s, 0.0, eta, ybar0, kd)))
            for eta in (0.0, 0.5, 1.0, 1.5):                                  # F8
                row += list(vols(cut(y8[eta]), F20 + 1e-4 * np.array(F8_OFF), 20.0)[0])
            y = cut(y11)                                                      # F11
            row += list(vols(y, np.mean(y) + SM_OFF, 20.0)[0])
            row += list(np.asarray(dec["market", anti](fit.sigma, fit.b, 0.0, ybar12, kd)))   # F12
            out[anti].append(row)
        if r % 16 == 15:
            print(f"replicate {r+1}/{R}, {time.perf_counter()-t0:.0f}s", flush=True)
    np.savez(NPZ, plain=np.array(out[False]), antithetic=np.array(out[True]), n_rep=N_REP,
             labels=np.array(["|".join(map(str, q[:3])) for q in Q]))


# ----------------------------------------------------------------------------- analysis
def blocks(structure):
    """Mask of covariance entries kept under a seed-sharing structure."""
    key = {"A": lambda q: 0, "C": lambda q: q[0], "B": lambda q: (q[0], q[1])}[structure]
    k = [key(q) for q in Q]
    return np.array([[a == b for b in k] for a in k])


def analyse():
    z = np.load(NPZ)
    paper = np.array([q[3] for q in Q])
    ok = ~np.isnan(paper)                     # points hidden in the figure (lines crossing) are not data
    held = np.array([q[4] if o else "missing" for q, o in zip(Q, ok)], dtype=object)
    cal = np.array([h is None for h in held])
    report = {}
    for est, n_paths in (("plain", int(z["n_rep"]) // 2), ("antithetic", int(z["n_rep"]))):
        X = z[est]
        R = X.shape[0]
        mu, S = X.mean(0), np.cov(X, rowvar=False)
        d = paper - mu
        d = np.where(ok, d, 0.0)                    # unused (excluded from every set)
        D = DIGITISATION**2 * np.eye(len(Q))
        print(f"\n=== {est} estimator: {R} replicates of {n_paths} paths ===")
        print("per point: paper - precise (bp), and the single-run s.d. at 65,536 paths")
        for q, di, s in zip(Q, d, np.sqrt(np.diag(S) * n_paths / 65536)):
            print(f"  {q[0]:5s} {q[1]:9s} {q[2]:5s} {di:+7.2f}  s.d.(64k) {s:5.2f}  {q[4] or ''}")
        res = {}
        for st in ("A", "C", "B"):
            S1 = S * n_paths * blocks(st)
            cov = lambda N: S1 / N + (S * blocks(st)) / R + D

            def nll(logN, idx):
                c = cov(np.exp(logN))[np.ix_(idx, idx)]
                sign, ld = np.linalg.slogdet(c)
                return 0.5 * (ld + d[idx] @ np.linalg.solve(c, d[idx]))

            def fit(idx):
                grid = np.linspace(np.log(1e2), np.log(1e8), 241)
                v = np.array([nll(g, idx) for g in grid])
                g0 = grid[np.argmin(v)]
                o = optimize.minimize_scalar(lambda g: nll(g, idx), bounds=(g0 - 0.1, g0 + 0.1), method="bounded")
                inside = grid[v - min(o.fun, v.min()) <= stats.chi2.ppf(0.95, 1) / 2]
                return float(np.exp(o.x)), float(np.exp(inside.min())), float(np.exp(inside.max())), o.fun

            ic = np.where(cal)[0]
            N, lo, hi, f = fit(ic)
            c = cov(N)[np.ix_(ic, ic)]
            chi2 = float(d[ic] @ np.linalg.solve(c, d[ic]))
            p_fit = float(stats.chi2.sf(chi2, len(ic) - 1))
            per_fig = {}
            for fg in sorted({q[0] for q, cl in zip(Q, cal) if cl}):
                idx = np.array([i for i in ic if Q[i][0] == fg])
                per_fig[fg] = fit(idx)[:3]

            def held_test(g, N):
                ih = np.where(held == g)[0]
                C = cov(N)
                Chc = C[np.ix_(ih, ic)]
                m = Chc @ np.linalg.solve(C[np.ix_(ic, ic)], d[ic])
                V = C[np.ix_(ih, ih)] - Chc @ np.linalg.solve(C[np.ix_(ic, ic)], Chc.T)
                r = d[ih] - m
                x2 = float(r @ np.linalg.solve(V, r))
                zs = r / np.sqrt(np.diag(V))
                return float(stats.chi2.sf(x2, len(ih))), [float(v) for v in zs], [float(v) for v in m]

            held_res = {}
            for g in ("A3", "A4", "A2"):
                p_hat, zs, m = held_test(g, N)
                p_lo, p_hi = held_test(g, lo)[0], held_test(g, hi)[0]
                held_res[g] = {"p": p_hat, "p_at_interval_ends": [p_lo, p_hi], "z": zs, "conditional_mean": m}
            res[st] = {"N": N, "N_95": [lo, hi], "neg_loglik": f, "chi2": chi2, "dof": len(ic) - 1,
                       "p_fit": p_fit, "per_figure_N": per_fig, "held_out": held_res}
            print(f"\nstructure {st}: N = {N:,.0f} (95%: {lo:,.0f} - {hi:,.0f}), -logL {f:.2f}, "
                  f"calibration chi2 {chi2:.1f} on {len(ic)-1} dof (p = {p_fit:.3f})")
            for fg, (n_, l_, h_) in per_fig.items():
                print(f"   {fg:5s} N = {n_:>12,.0f}  ({l_:,.0f} - {h_:,.0f})")
            for g, h in held_res.items():
                ih = np.where(held == g)[0]
                print(f"   held out {g}: p = {h['p']:.3g} (at interval ends {h['p_at_interval_ends'][0]:.3g}, "
                      f"{h['p_at_interval_ends'][1]:.3g}); z by point: "
                      + ", ".join(f"{Q[i][0]} {Q[i][1]} {Q[i][2]} {zz:+.1f}" for i, zz in zip(ih, h["z"])))
        report[est] = {"n_paths": n_paths, "replicates": R, "d": d.tolist(),
                       "sd_per_path": np.sqrt(np.diag(S) * n_paths).tolist(), "structures": res}
    report["quantities"] = [list(q) for q in Q]
    (RES / "a3_paper_path_count.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    if "--analyse" not in sys.argv:
        simulate_replicates(int(sys.argv[1]) if len(sys.argv) > 1 else 512)
    analyse()

"""A1/A9: Figure 4 under every candidate reading of the yield option.

For each measure (P undiscounted, P discounted pathwise, Q^T [= spot Q with
discounting], Q^{T+delta}) and each inversion forward (the model's own mean in
that measure, or the curve forward Y(0,T,T+delta) where the paper draws its
ATM line), price the paper's strike grid (forward + 40k bp) at b in
{0,...,1}, sigma = 1%, bootstrap scaling, and compare with the digitised
Figure 4. Also checks the helper against the library pricer and two
model-free identities.
"""
import sys
import numpy as np
import jax
from common import BP, terminal_yield, smile, save
from slidingcurve import Model, markets, volscale, pricing as pr

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2**16
curve = markets.paper_curve("baseline")
fig4 = markets.paper_figures()["fig4"]
base = Model(kappa=0.03, delta=5.0, sigma=0.01)
grid = np.arange(1.0, 21.0)
key = jax.random.PRNGKey(0)
SKEWS = (0.0, 0.25, 0.5, 0.75, 1.0)

# --- checks of the helper ----------------------------------------------------
m = base._replace(b=0.5)
fn = volscale.bootstrap_fn(curve, m, grid)
y_fwd, _ = terminal_yield(m, curve, 20.0, fn, key, 25.0, N)
mean_lib, p_lib, _ = pr.yield_option_prices(m, curve, 20.0, np.array([0.0]), fn, key, N)
y_q, d_q = terminal_yield(m, curve, 20.0, fn, key, "Q", N)
y_t, _ = terminal_yield(m, curve, 20.0, fn, key, 20.0, N)
y_p, _ = terminal_yield(m, curve, 20.0, fn, key, "P", N)
se = lambda a: a.std() / np.sqrt(len(a)) * BP
print(f"helper vs library, Q^(T+d) mean: {BP*y_fwd.mean():.3f} vs {BP*float(mean_lib):.3f}bp")
print(f"E^Q[D Y]/E^Q[D] = {BP*np.mean(d_q*y_q)/np.mean(d_q):.2f}bp vs E^T[Y] = {BP*y_t.mean():.2f} +- {se(y_t):.2f}bp;"
      f"  E^Q[D]/P(0,T) = {np.mean(d_q)/float(curve.discount(20.0)):.5f}")
print(f"E^P[Y1(20)] = {BP*y_p.mean():.2f} +- {se(y_p):.2f}bp vs Y1(0) = {BP*float(curve.fwd_yield(0.0, 5.0)):.2f}bp")

# --- the variants ------------------------------------------------------------
VARIANTS = [("Q^(T+d)", 1, "model mean"), ("Q^(T+d)", 1, "curve fwd"),
            ("Q^T", 0, "model mean"), ("Q^T", 0, "curve fwd"),
            ("P", None, "model mean"), ("P", None, "curve fwd"), ("P disc", None, "curve fwd")]
results = {}
for T in (5, 10, 15, 20):
    r = fig4[str(T)]
    F = float(curve.fwd_yield(T, T + 5.0))
    K = F + 1e-4 * np.asarray(r["offsets"])
    for b in SKEWS:
        m = base._replace(b=b)
        fn = volscale.bootstrap_fn(curve, m, grid)
        sims = {"Q^(T+d)": terminal_yield(m, curve, T, fn, key, T + 5.0, N),
                "Q^T": terminal_yield(m, curve, T, fn, key, float(T), N),
                "P": terminal_yield(m, curve, T, fn, key, "P", N)}
        for meas, _, fwd in VARIANTS:
            y1, disc = sims["P" if meas == "P disc" else meas]
            Finv = y1.mean() if fwd == "model mean" else F
            v = smile(y1, K, Finv, T, disc if meas == "P disc" else None)
            results.setdefault(f"{meas} / {fwd}", {})[f"{T}|{b}"] = (BP * v).tolist()

summary = {}
print("\nmean |model - paper| (bp) over each panel; [missing points: model / paper]")
for name, res in results.items():
    errs, row = [], []
    for T in (5, 10, 15, 20):
        e, miss_m, miss_p = [], 0, 0
        for b in SKEWS:
            mine = np.array(res[f"{T}|{b}"], dtype=float)
            paper = np.array(fig4[str(T)][f"b={b:.2f}"], dtype=float)
            ok = ~np.isnan(mine) & ~np.isnan(paper)
            e.extend(np.abs(mine - paper)[ok]); miss_m += np.isnan(mine).sum(); miss_p += np.isnan(paper).sum()
        row.append(f"{T}y {np.mean(e):5.1f} [{miss_m:2d}/{miss_p:2d}]")
        errs.extend(e)
    summary[name] = float(np.mean(errs))
    print(f"{name:26s} " + "  ".join(row) + f"   all {np.mean(errs):5.2f}")
save("a1_conventions", {"n_paths": N, "smiles": results, "mean_abs_err": summary})

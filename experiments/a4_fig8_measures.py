"""A4: Figure 8 (20y x 5y smiles by eta at sigma = 1%, b = 0) in each measure,
on the paper's absolute strike grid (387 + 50k bp), with the self-consistent
inversion forward (the model mean in that measure) and with the curve
forward 387bp. Reports the level, the position of the smile minimum and the
error against the digitised figure. Also estimates the Monte-Carlo spread of
the eta = 0 level for a small path count, to test whether the paper's
71.4-71.5bp (exact: 70.86bp) is sampling error."""
import numpy as np
import jax
from common import BP, terminal_yield, smile, save
from slidingcurve import Model, markets, volscale

N = 2**17
curve = markets.paper_curve("baseline")
fig8 = markets.paper_figures()["fig8"]
base = Model(kappa=0.03, delta=5.0, sigma=0.01, kappa_z=0.1)
ybar = lambda t: volscale.ybar_zeroth(curve, 5.0, t)
F = float(curve.fwd_yield(20.0, 25.0))
K = 387e-4 + 1e-4 * np.asarray(fig8["offsets"])
fine = np.linspace(K[0], K[-1], 201)


def minimum(y1, Finv):
    v = smile(y1, fine, Finv, 20.0)
    return BP * fine[np.nanargmin(v)]


out = {}
paper_min = {}
for e in (0.5, 1.0, 1.5):
    p = np.array(fig8[f"eta={e}"]); i = int(np.argmin(p))
    a, b_, c = p[i - 1], p[i], p[i + 1]
    paper_min[e] = 387 + 50 * float(np.asarray(fig8["offsets"])[i] / 50) + 50 * (a - c) / (2 * (a + c - 2 * b_))
print("paper minimum (parabola through the three lowest points):", {k: round(v, 1) for k, v in paper_min.items()})
for meas in ("Q^(T+d)", "Q^T", "P"):
    m_arg = {"Q^(T+d)": 25.0, "Q^T": 20.0, "P": "P"}[meas]
    for fwd in ("model mean", "curve fwd"):
        row = {}
        for e in (0.0, 0.5, 1.0, 1.5):
            y1, _ = terminal_yield(base._replace(eta=e), curve, 20.0, ybar, jax.random.PRNGKey(0), m_arg, N)
            Finv = y1.mean() if fwd == "model mean" else F
            v = BP * smile(y1, K, Finv, 20.0)
            row[e] = {"vols": v.tolist(), "err": float(np.nanmean(np.abs(v - np.array(fig8[f'eta={e}'])))),
                      "atm387": float(v[5]), "min": minimum(y1, Finv) if e > 0 else None,
                      "mean": BP * float(y1.mean())}
        out[f"{meas} / {fwd}"] = row
        print(f"{meas:8s} / {fwd:10s}: mean Y {row[1.5]['mean']:6.1f}bp | " + " | ".join(
            f"eta {e}: v(387) {row[e]['atm387']:5.2f} err {row[e]['err']:4.2f}" +
            (f" min@{row[e]['min']:5.0f}" if e > 0 else "") for e in (0.0, 0.5, 1.0, 1.5)), flush=True)

# sampling spread of the eta = 0 level: 64 independent runs at a small path count, no antithetics
spread = {}
for n in (2**12, 2**14):
    lev = []
    for s in range(64):
        z = np.random.default_rng(s).standard_normal(n)
        y1 = F + 70.86e-4 * np.sqrt(20.0) * z
        lev.append(BP * smile(y1, K[5:6], y1.mean(), 20.0)[0])
    spread[n] = float(np.std(lev))
    print(f"Gaussian 20y ATM vol from {n} plain paths: spread {np.std(lev):.2f}bp; "
          f"P(|error| >= 0.55bp) = {np.mean(np.abs(np.array(lev) - 70.86) >= 0.55):.2f}")
save("a4_fig8_measures", {"variants": out, "paper_min": paper_min, "eta0_spread": spread})

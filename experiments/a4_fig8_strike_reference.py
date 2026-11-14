"""A4/A9: Figure 8 with strikes referenced to the model mean versus the curve
forward (Q^(T+delta), inversion around the model mean in both cases)."""
import numpy as np
import jax
from common import BP, terminal_yield, smile, save
from slidingcurve import Model, markets, volscale

N = 2**17
curve = markets.paper_curve("baseline")
fig8 = markets.paper_figures()["fig8"]
off = 1e-4 * np.asarray(fig8["offsets"])
base = Model(kappa=0.03, delta=5.0, sigma=0.01, kappa_z=0.1)
ybar = lambda t: volscale.ybar_zeroth(curve, 5.0, t)
F = float(curve.fwd_yield(20.0, 25.0))
out = {}
for ref in ("mean", "forward"):
    row = {}
    for e in (0.0, 0.5, 1.0, 1.5):
        y1, _ = terminal_yield(base._replace(eta=e), curve, 20.0, ybar, jax.random.PRNGKey(0), 25.0, N)
        K = (y1.mean() if ref == "mean" else F) + off
        v = BP * smile(y1, K, y1.mean(), 20.0)
        paper = np.array(fig8[f"eta={e}"])
        row[e] = {"vols": v.tolist(), "mean_abs_err": float(np.mean(np.abs(v - paper))),
                  "argmin_offset_bp": float(BP * off[int(np.argmin(v))]), "paper_argmin_offset_bp": float(BP * off[int(np.argmin(paper))])}
    out[ref] = row
    print(f"strikes at {ref:7s}: " + " | ".join(
        f"eta {e}: err {row[e]['mean_abs_err']:.2f}bp, lowest point at {row[e]['argmin_offset_bp']:+.0f} (paper {row[e]['paper_argmin_offset_bp']:+.0f})"
        for e in (0.0, 0.5, 1.0, 1.5)))
save("a4_fig8_strike_reference", out)

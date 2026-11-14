"""A9: scan the measure (payment date tau in [T, T + delta]) with strikes
referenced to the model mean, against Figure 8 (b = 0, eta > 0, baseline
curve, sigma = 1%) and Figure 11's fitted vols at the paper's stated
sigma = 1.1999%, b = 0.367 (market curve). Also the annuity measure of a
5y annual-pay swap, as a mixture over its payment dates."""
import numpy as np
import jax
from common import BP, terminal_yield, smile, save
from slidingcurve import Model, markets, volscale

N = 2**17
key = jax.random.PRNGKey(0)
fig8 = markets.paper_figures()["fig8"]
off8 = 1e-4 * np.asarray(fig8["offsets"])
base_c, mkt_c = markets.paper_curve("baseline"), markets.paper_curve("market")
offs11, market, paper_fit = markets.paper_smile()
m11 = Model(kappa=0.03, delta=5.0, sigma=0.011999, b=0.367)
yb11 = volscale.bootstrap_fn(mkt_c, m11, np.arange(1.0, 21.0))
yb8 = lambda t: volscale.ybar_zeroth(base_c, 5.0, t)
out = {}
for tau_off in (0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0):
    tau = 20.0 + tau_off
    e8 = []
    for e in (0.5, 1.0, 1.5):
        y1, _ = terminal_yield(Model(kappa=0.03, delta=5.0, sigma=0.01, eta=e), base_c, 20.0, yb8, key, tau, N)
        v = BP * smile(y1, y1.mean() + off8, y1.mean(), 20.0)
        e8.append((float(np.mean(np.abs(v - np.array(fig8[f"eta={e}"])))), float(BP * off8[int(np.argmin(v))])))
    y1, _ = terminal_yield(m11, mkt_c, 20.0, yb11, key, tau, N)
    v11 = BP * smile(y1, y1.mean() + offs11, y1.mean(), 20.0)
    d11 = v11 - BP * paper_fit
    out[tau_off] = {"fig8": e8, "fig11_minus_paper_fit": d11.tolist()}
    print(f"tau = T + {tau_off:3.1f}: Figure 8 err/lowest point " +
          ", ".join(f"{a:.2f}bp@{b:+.0f}" for a, b in e8) + f" | Figure 11 minus paper fit {np.round(d11, 2)}", flush=True)
save("a9_measure_scan", out)

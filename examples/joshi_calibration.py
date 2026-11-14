"""End to end on a published test market: calibrate to a swaption smile, then
decompose the yield curve (the workflow of Section 9.4, different data).

Market: Mark Joshi's displaced-diffusion LMM example (see markets.joshi_2014).
Instrument: the 10y x 5y payer swaption at five strikes. The paper uses
20y x 5y; this curve ends at 20.5 years, so the expiry is shortened.

    python examples/joshi_calibration.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, calibration, decompose, markets, simulate, time_grid, volscale

BP = 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    mkt = markets.joshi_2014()
    quotes = mkt.quotes(expiry=10.0, tenor=5.0, strike_offsets=[-0.01, -0.005, 0.0, 0.005, 0.01])
    atm = quotes._replace(strikes=quotes.strikes[2:3], prices=quotes.prices[2:3],
                          normal_vols=quotes.normal_vols[2:3])
    start = Model(kappa=0.03, delta=5.0, sigma=0.01)

    fits = {"ATM only (sigma)": calibration.calibrate(start, mkt.curve, atm, free=("sigma",)),
            "smile (sigma, b)": calibration.calibrate(start, mkt.curve, quotes, free=("sigma", "b"))}
    print(f"forward swap rate {100*quotes.forward:.3f}%   market normal vols (bp): "
          + " ".join(f"{BP*v:.1f}" for v in quotes.normal_vols))

    t = time_grid(15.0, steps_per_year=12)
    mats = jnp.arange(1.0, 16.0)
    grid = np.arange(1.0, 16.0)
    decs = {}
    for name, fit in fits.items():
        m = fit.model
        print(f"\n{name}: sigma = {100*m.sigma:.4f}%  b = {m.b:.3f}   "
              f"max |vol error| = {np.max(np.abs(fit.vol_errors_bp)):.2f}bp "
              f"(MC s.e. {np.max(fit.price_se_bp):.2f}bp)")
        ybar = volscale.bootstrap_fn(mkt.curve, m, grid)(t)
        d = decompose(simulate(m, mkt.curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**16),
                      m, mkt.curve, mats)
        decs[name] = d
        print(f"{'T':>4} {'Y':>8} {'Y_eh':>8} {'theta':>8} {'gamma':>8} {'resid':>7}  (bp)")
        for j in (0, 4, 9, 14):
            print(f"{float(d.T[j]):4.0f} {BP*d.y[j]:8.1f} {BP*d.y_eh[j]:8.1f} "
                  f"{BP*d.theta[j]:8.1f} {BP*d.gamma[j]:8.1f} {BP*d.residual[j]:7.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(BP * (quotes.strikes - quotes.forward), BP * quotes.normal_vols, "ko", label="market (Joshi DD-LMM)")
    a.plot(BP * (quotes.strikes - quotes.forward), BP * fits["smile (sigma, b)"].model_vols,
           label="model, sigma and b fitted")
    a.set(xlabel="strike minus forward (bp)", ylabel="implied normal vol (bp)",
          title="10y x 5y swaption smile")
    a.legend(frameon=False)
    for name, d in decs.items():
        b.plot(d.T, BP * d.gamma, label=f"gamma, {name}")
        b.plot(d.T, BP * d.theta, "--", label=f"theta, {name}")
    b.set(xlabel="maturity T (years)", ylabel="basis points", title="Term premium and convexity")
    b.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "joshi_calibration.png", dpi=150)
    print(f"\nfigure written to {out}/joshi_calibration.png")


if __name__ == "__main__":
    main()

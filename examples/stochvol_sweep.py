"""Impact of stochastic volatility on convexity (Section 9.3, Figures 7-9).

Skew b = 0, variance mean reversion kappa_z = 0.1, eta in {0, 0.5, 1, 1.5}.

  left    gamma(T) with sigma left at its eta = 0 value            (Figure 7)
  middle  20y x 5y normal-volatility smile at that fixed sigma     (Figure 8)
  right   gamma(T) with sigma recalibrated at each eta so that the
          20y x 5y at-the-money volatility is unchanged            (Figure 9)

The benchmark option is the caplet on the 5y simply compounded rate expiring
in 20 years. The curve is an illustrative Nelson-Siegel curve, NOT the
paper's synthetic curve, so shapes and ratios are comparable and levels are
not.

    python examples/stochvol_sweep.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import (Curve, Model, bachelier as bl, calibration, decompose, markets,
                          pricing as pr, simulate, time_grid, volscale)

BP = 1e4
ETAS = (0.0, 0.5, 1.0, 1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**17)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
    base = Model(kappa=0.03, delta=5.0, sigma=0.01, kappa_z=0.1)
    ybar_fn = lambda t: volscale.ybar_zeroth(curve, base.delta, t)   # unused at b = 0
    swp = pr.Swaption.caplet(20.0, base.delta)
    fwd, ann = float(pr.forward_swap_rate(curve, swp)), float(pr.annuity(curve, swp))
    strikes = fwd + np.linspace(-0.02, 0.02, 9)
    pkey, skey = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

    def smile(model):
        p, _ = pr.swaption_prices(model, curve, swp, strikes, ybar_fn, pkey, n_paths=args.paths)
        return bl.implied_normal_vol(np.asarray(p) / ann, fwd, strikes, swp.expiry)

    t = time_grid(40.0, steps_per_year=12)
    mats = jnp.arange(1.0, 41.0)

    def convexity(model):
        prof = simulate(model, curve, t, ybar_fn(t), skey, n_paths=args.paths)
        return decompose(prof, model, curve, mats)

    atm0 = smile(base)[4]
    target = markets.Quotes(swp, fwd, ann, strikes[4:5],
                            np.array([ann * float(bl.normal_price(fwd, fwd, atm0, swp.expiry))]),
                            np.array([atm0]))
    rows = []
    for eta in ETAS:
        fixed = base._replace(eta=eta)
        recal = calibration.calibrate(fixed, curve, target, free=("sigma",), ybar_fn=ybar_fn,
                                      key=pkey, n_paths=args.paths).model
        rows.append((eta, smile(fixed), convexity(fixed), recal, convexity(recal)))

    g0 = rows[0][2].gamma
    print(f"20y x 5y forward {100*fwd:.3f}%, ATM normal vol at eta = 0: {BP*atm0:.2f}bp\n")
    print(" eta | ATM vol, sigma fixed | recalibrated sigma | gamma(40y) fixed | gamma(40y) recal. | "
          "recal./eta=0 at 10y, 20y, 40y")
    for eta, sm, dfix, recal, drec in rows:
        r = np.asarray(drec.gamma / g0)
        print(f" {eta:3.1f} | {BP*sm[4]:17.2f}bp | {100*recal.sigma:17.4f}% | "
              f"{BP*dfix.gamma[-1]:7.1f} +- {BP*dfix.gamma_se[-1]:4.1f}bp | "
              f"{BP*drec.gamma[-1]:8.1f} +- {BP*drec.gamma_se[-1]:4.1f}bp | "
              f"{r[9]:.3f}  {r[19]:.3f}  {r[39]:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))
    for eta, sm, dfix, recal, drec in rows:
        a.plot(dfix.T, BP * dfix.gamma, label=f"eta = {eta}")
        b.plot(BP * (strikes - fwd), BP * sm, label=f"eta = {eta}")
        c.plot(drec.T, BP * drec.gamma, label=f"eta = {eta}, sigma = {100*recal.sigma:.3f}%")
    a.set(xlabel="maturity T (years)", ylabel="basis points",
          title="Convexity, sigma fixed (cf. Figure 7)")
    b.set(xlabel="strike minus forward (bp)", ylabel="implied normal vol (bp)",
          title="20y x 5y smile, sigma fixed (cf. Figure 8)")
    c.set(xlabel="maturity T (years)", ylabel="basis points",
          title="Convexity, ATM vol fixed (cf. Figure 9)")
    c.set_ylim(a.get_ylim()[0], max(c.get_ylim()[1], a.get_ylim()[1]))
    a.set_ylim(c.get_ylim())
    for ax in (a, b, c):
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "stochvol_sweep.png", dpi=150)
    print(f"\nfigure written to {out}/stochvol_sweep.png")


if __name__ == "__main__":
    main()

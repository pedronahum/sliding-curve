"""Impact of stochastic volatility on convexity (Section 9.3, Figures 7-9),
against the paper's own plotted results.

Baseline curve, b = 0, kappa_z = 0.1, eta in {0, 0.5, 1, 1.5}.

  left    gamma(T) with sigma left at its eta = 0 value            (Figure 7)
  middle  20y x 5y yield-option smile at that fixed sigma          (Figure 8)
  right   gamma(T) with sigma recalibrated at each eta so that the
          20y x 5y at-the-money volatility is unchanged            (Figure 9)

    python examples/stochvol_sweep.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import (Model, bachelier as bl, calibration, decompose, markets,
                          pricing as pr, simulate, time_grid, volscale)

BP = 1e4
ETAS = (0.0, 0.5, 1.0, 1.5)
EXPIRY = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = markets.paper_curve("baseline")
    figs = markets.paper_figures()
    base = Model(kappa=0.03, delta=5.0, sigma=0.01, kappa_z=0.1)
    ybar_fn = lambda t: volscale.ybar_zeroth(curve, base.delta, t)   # unused at b = 0
    offsets = 1e-4 * np.asarray(figs["fig8"]["offsets"])
    pkey, skey = jax.random.PRNGKey(0), jax.random.PRNGKey(1)

    def smile(model):
        mean, p, _ = pr.yield_option_prices(model, curve, EXPIRY, offsets, ybar_fn, pkey,
                                            n_paths=args.paths)
        smile.mean = float(mean)          # strikes are mean + offset; the paper's are 387bp + offset
        return bl.implied_normal_vol(np.asarray(p), 0.0, offsets, EXPIRY)

    t = time_grid(40.0, steps_per_year=12)
    mats = jnp.arange(1.0, 41.0)

    def convexity(model):
        prof = simulate(model, curve, t, ybar_fn(t), skey, n_paths=args.paths)
        return decompose(prof, model, curve, mats)

    atm0 = smile(base)[5]
    rows = []
    for eta in ETAS:
        fixed = base._replace(eta=eta)
        recal = calibration.calibrate_yield_smile(fixed, curve, EXPIRY, offsets[5:6], [atm0],
                                                  free=("sigma",), key=pkey, n_paths=args.paths).model
        rows.append((eta, smile(fixed), convexity(fixed), recal, convexity(recal), smile.mean))

    ref9, ref8 = figs["fig9"], figs["fig8"]
    print(" eta | ATM vol, sigma fixed (paper) | recalibrated sigma | gamma(40y) recalibrated (paper) | "
          "gamma(40y) sigma fixed")
    for eta, sm, dfix, recal, drec, _ in rows:
        print(f" {eta:3.1f} | {BP*sm[5]:8.2f}bp ({ref8[f'eta={eta}'][5]:6.2f}bp)     | "
              f"{100*recal.sigma:17.4f}% | {BP*drec.gamma[-1]:7.1f} +- {BP*drec.gamma_se[-1]:3.1f}bp "
              f"({ref9[f'eta={eta}'][-1]:6.1f}bp)      | {BP*dfix.gamma[-1]:7.1f}bp")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16, 4.5))
    ref_t = np.asarray(ref9["T"])
    for i, (eta, sm, dfix, recal, drec, mean) in enumerate(rows):
        a.plot(dfix.T, BP * dfix.gamma, color=f"C{i}", label=f"eta = {eta}")
        # absolute strikes: this code's are offsets from the yield's mean, the
        # paper's from its at-the-money line at the curve forward (387bp)
        b.plot(BP * (mean + offsets), BP * sm, color=f"C{i}", label=f"eta = {eta}")
        b.plot(387.0 + BP * offsets, np.array(ref8[f"eta={eta}"], dtype=float), "o", ms=3.5,
               color=f"C{i}", markerfacecolor="white")
        c.plot(drec.T, BP * drec.gamma, color=f"C{i}", label=f"eta = {eta}, sigma = {100*recal.sigma:.3f}%")
        c.plot(ref_t[11::12], np.array(ref9[f"eta={eta}"], dtype=float)[11::12], "o", ms=3.5,
               color=f"C{i}", markerfacecolor="white")
    for ax in (b, c):
        ax.plot([], [], "o", ms=3.5, color="grey", markerfacecolor="white", label="paper (read off figure)")
    a.set(xlabel="maturity T (years)", ylabel="basis points",
          title="Convexity, sigma fixed (cf. Figure 7)")
    b.set(xlabel="strike (bp)", ylabel="implied normal vol (bp)",
          title="20y x 5y smile, sigma fixed (cf. Figure 8)")
    c.set(xlabel="maturity T (years)", ylabel="basis points",
          title="Convexity, ATM vol fixed (cf. Figure 9)")
    a.set_ylim(c.get_ylim())
    for ax in (a, b, c):
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "stochvol_sweep.png", dpi=150)
    print(f"\nfigure written to {out}/stochvol_sweep.png")


if __name__ == "__main__":
    main()

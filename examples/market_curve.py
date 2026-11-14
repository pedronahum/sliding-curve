"""Section 9.4: the 21-Aug-2026 USD curve and 20y x 5y smile (Figures 10-12).

The curve, the five smile quotes and the paper's results are read off its
figures (markets.paper_curve('market'), markets.paper_smile). kappa = 0.03,
delta = 5, eta = 0 unless stated.

    python examples/market_curve.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, calibration, decompose, markets, simulate, time_grid, volscale

BP = 1e4
EXPIRY = 20.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**15)
    ap.add_argument("--with-eta", action="store_true", help="also run the three-parameter fit (several minutes)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = markets.paper_curve("market")
    offsets, market, paper_fit = markets.paper_smile()
    figs = markets.paper_figures()
    start = Model(kappa=0.03, delta=5.0, sigma=0.01)
    kw = dict(n_paths=args.paths)

    atm = calibration.calibrate_yield_smile(start, curve, EXPIRY, offsets[2:3], market[2:3],
                                            free=("sigma",), **kw)
    # the paper's fit matches the at-the-money quote exactly: pin it with a large weight
    skew = calibration.calibrate_yield_smile(start, curve, EXPIRY, offsets, market,
                                             free=("sigma", "b"), weights=[1, 1, 30, 1, 1], **kw)
    print("market vols (bp):    " + " ".join(f"{BP*v:6.1f}" for v in market))
    print("paper's fit (bp):    " + " ".join(f"{BP*v:6.1f}" for v in paper_fit))
    print("this code's fit (bp):" + " ".join(f"{BP*v:6.1f}" for v in skew.model_vols))
    print(f"\nATM only:  sigma = {100*atm.model.sigma:.3f}%            (paper: 1.157%)")
    print(f"with skew: sigma = {100*skew.model.sigma:.3f}%  b = {skew.model.b:.3f}"
          "   (paper: b = 0.367; sigma = 1.12% in the text, 1.1999% in the caption of Figure 13)")
    if args.with_eta:
        # start from the skew fit: with eta free there is a second, spurious basin
        full = calibration.calibrate_yield_smile(skew.model._replace(eta=0.5), curve, EXPIRY,
                                                 offsets, market, free=("sigma", "b", "eta"), **kw)
        print(f"three parameters: sigma = {100*full.model.sigma:.3f}%  b = {full.model.b:.3f}  "
              f"eta = {full.model.eta:.3f}   (paper: 1.492%, 0.417, 0.779)   "
              f"max error {np.max(np.abs(full.vol_errors_bp)):.2f}bp")

    t = time_grid(30.0, steps_per_year=12)
    mats = jnp.arange(1.0, 31.0)
    grid = np.arange(1.0, 31.0)
    decs = {}
    # Figure 12 at the paper's own parameters separates the engine from the fit
    stated = atm.model._replace(sigma=0.011999, b=0.367)
    for name, m, fig in (("b = 0 (Figure 10)", atm.model, "fig10"),
                         ("skew, this code's fit (Figure 12)", skew.model, "fig12"),
                         ("skew, paper's sigma = 1.1999%, b = 0.367 (Figure 12)", stated, "fig12")):
        ybar = volscale.bootstrap_fn(curve, m, grid)(t)
        d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=2**16),
                      m, curve, mats)
        ref = figs[fig]
        decs[name] = (d, ref)
        print(f"\n{name}\n   T |   Y_eh  theta  gamma |  paper: Y_eh  theta  gamma   (bp)")
        for T in (5, 10, 20, 30):
            j, r = T - 1, 4 * T - 1
            print(f"{T:4d} | {BP*d.y_eh[j]:6.1f} {BP*d.theta[j]:6.1f} {BP*d.gamma[j]:6.1f} |"
                  f"        {ref['Y_eh'][r]:6.1f} {ref['theta'][r]:6.1f} {ref['gamma'][r]:6.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    k = BP * offsets
    axes[0].plot(k, BP * market, "ko-", label="market (read off Figure 11)")
    axes[0].plot(k, BP * paper_fit, "s--", color="C3", label="paper's fit")
    axes[0].plot(k, BP * skew.model_vols, "^:", color="C0", label="this code's fit")
    axes[0].set(xlabel="strike minus at-the-money (bp)", ylabel="normal vol (bp)",
                title="20y x 5y smile (cf. Figure 11)")
    axes[0].legend(frameon=False)
    shown = [k for k in decs if "this code" not in k]
    for ax, name in zip(axes[1:], shown):
        d, ref = decs[name]
        rt = np.asarray(ref["T"])[7::8]
        for i, (label, mine, key) in enumerate((("Y_eh - Y_eh(1y)", d.y_eh - d.y_eh[0], "Y_eh"),
                                                ("theta", d.theta, "theta"), ("gamma", d.gamma, "gamma"))):
            r = np.array(ref[key], dtype=float)
            r = r - r[3] if key == "Y_eh" else r
            ax.plot(d.T, BP * mine, color=f"C{i}", label=label)
            ax.plot(rt, r[7::8], "o", ms=3.5, color=f"C{i}", markerfacecolor="white")
        ax.plot([], [], "o", ms=3.5, color="grey", markerfacecolor="white", label="paper (read off figure)")
        ax.set(xlabel="maturity T (years)", ylabel="basis points", title=name.replace("skew, paper's ", "skew: "))
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "market_curve.png", dpi=150)
    print(f"\nfigure written to {out}/market_curve.png")


if __name__ == "__main__":
    main()

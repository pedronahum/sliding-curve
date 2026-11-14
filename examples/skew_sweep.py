"""Impact of skew on convexity (Section 9.2, Figures 3 and 6), against the
paper's own plotted results.

Curves and reference values are read off the paper's figures (see
markets.paper_curve and tools/digitise_figures.py). Settings follow Section
9.1: delta = 5, kappa = 0.03, sigma = 1%, eta = 0, unweighted bootstrap for
the volatility scaling (annual expiry grid, which the paper does not state).

    python examples/skew_sweep.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, decompose, markets, pricing as pr, simulate, time_grid, volscale

BP = 1e4
SKEWS = (0.0, 0.25, 0.5, 0.75, 1.0)
CASES = {"baseline": ("fig2", "fig3", "Baseline curve (cf. Figure 3)"),
         "steep": ("fig5", "fig6", "Upward-sloping curve (cf. Figure 6)")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    figs = markets.paper_figures()
    base = Model(kappa=0.03, delta=5.0, sigma=0.01)
    t = time_grid(40.0, steps_per_year=12)
    mats = jnp.arange(1.0, 41.0)
    grid = np.arange(1.0, 41.0)
    show = [9, 19, 29, 39]                      # 10, 20, 30, 40 years
    results = {}

    for which, (fig_y, fig_g, title) in CASES.items():
        curve = markets.paper_curve(which)
        ref_t = np.asarray(figs[fig_g]["T"])
        print(f"\n{title}\n   b |  gamma at 10y, 20y, 30y, 40y (bp)   | paper, read off the figure      | s.e. at 40y")
        for b in SKEWS:
            m = base._replace(b=b)
            ybar = volscale.bootstrap_fn(curve, m, grid)(t)
            d = decompose(simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=args.paths),
                          m, curve, mats)
            ref = np.array(figs[fig_g][f"b={b:.2f}"], dtype=float)
            results[which, b] = (d, ref_t, ref)
            paper = [ref[np.argmin(np.abs(ref_t - T))] for T in (10, 20, 30, 40)]
            print(f"{b:4.2f} | " + " ".join(f"{BP*d.gamma[j]:7.1f}" for j in show) + "     | "
                  + " ".join(f"{p:7.1f}" for p in paper) + f" | {BP*d.gamma_se[-1]:.1f}")
            if b == 0.0:
                yeh = np.array(figs[fig_y]["Y_eh"], dtype=float)
                fig_t = np.asarray(figs[fig_y]["T"])
                err = [BP * float(d.y_eh[j]) - yeh[np.argmin(np.abs(fig_t - (j + 1)))] for j in range(40)]
                print(f"     | Y_eh versus the paper's figure: max difference {np.nanmax(np.abs(err)):.2f}bp")

        # what the volatility scaling holds fixed: ATM caplets on the 5y rate
        for expiry in (5.0, 20.0):
            swp = pr.Swaption.caplet(expiry, base.delta)
            k = jnp.array([float(pr.forward_swap_rate(curve, swp))])
            p0, p1 = (float(pr.swaption_prices(base._replace(b=b), curve, swp, k,
                                               volscale.bootstrap_fn(curve, base, grid),
                                               jax.random.PRNGKey(0), n_paths=args.paths)[0][0])
                      for b in (0.0, 1.0))
            print(f"     | ATM caplet, {expiry:.0f}y expiry: b = 1 price / b = 0 price = {p1/p0:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (which, (_, _, title)) in zip(axes, CASES.items()):
        for i, b in enumerate(SKEWS):
            d, ref_t, ref = results[which, b]
            line, = ax.plot(d.T, BP * d.gamma, color=f"C{i}", label=f"b = {b:.2f}")
            ax.plot(ref_t[11::12], ref[11::12], "o", ms=3.5, color=line.get_color(),
                    markerfacecolor="white")
        ax.plot([], [], "o", ms=3.5, color="grey", markerfacecolor="white",
                label="paper (read off figure)")
        ax.set(xlabel="maturity T (years)", ylabel="convexity gamma(T), basis points", title=title)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "skew_sweep.png", dpi=150)
    print(f"\nfigure written to {out}/skew_sweep.png")


if __name__ == "__main__":
    main()

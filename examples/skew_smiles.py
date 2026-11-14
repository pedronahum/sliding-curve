"""Yield-option smiles by skew at four expiries (Figure 4), against the
paper's plotted values. Baseline curve, sigma = 1%, eta = 0, bootstrap scaling.

    python examples/skew_smiles.py [--out out]
"""
import argparse
import pathlib

import jax
import numpy as np

from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale

BP = 1e4
SKEWS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = markets.paper_curve("baseline")
    ref = markets.paper_figures()["fig4"]
    base = Model(kappa=0.03, delta=5.0, sigma=0.01)
    grid = np.arange(1.0, 21.0)
    res = {}
    print("expiry |    b | ATM vol (paper) | vol at -80bp (paper) | vol at +80bp (paper) | mean abs diff")
    for expiry in (5, 10, 15, 20):
        r = ref[str(expiry)]
        offsets = 1e-4 * np.asarray(r["offsets"])
        for b in SKEWS:
            m = base._replace(b=b)
            _, p, _ = pr.yield_option_prices(m, curve, float(expiry), offsets,
                                             volscale.bootstrap_fn(curve, m, grid),
                                             jax.random.PRNGKey(0), n_paths=args.paths)
            vols = BP * bl.implied_normal_vol(np.asarray(p), 0.0, offsets, float(expiry))
            paper = np.array(r[f"b={b:.2f}"], dtype=float)
            res[expiry, b] = (BP * offsets, vols, paper)
            cell = lambda i: f"{vols[i]:6.1f} ({paper[i]:6.1f})"
            print(f"{expiry:5d}y | {b:4.2f} | {cell(5)} | {cell(3)}      | {cell(7)}      | "
                  f"{np.nanmean(np.abs(vols - paper)):.1f}bp")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    for ax, expiry in zip(axes.ravel(), (5, 10, 15, 20)):
        for i, b in enumerate(SKEWS):
            k, vols, paper = res[expiry, b]
            ax.plot(k, vols, color=f"C{i}", label=f"b = {b:.2f}")
            ax.plot(k, paper, "o", ms=3.5, color=f"C{i}", markerfacecolor="white")
        ax.plot([], [], "o", ms=3.5, color="grey", markerfacecolor="white", label="paper (read off figure)")
        ax.set(xlabel="strike minus at-the-money (bp)", ylabel="implied normal vol (bp)",
               title=f"{expiry}y x 5y")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Yield option implied normal vol (cf. Figure 4)")
    fig.tight_layout()
    fig.savefig(out / "skew_smiles.png", dpi=150)
    print(f"\nfigure written to {out}/skew_smiles.png")


if __name__ == "__main__":
    main()

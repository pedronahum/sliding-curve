"""Density of the short rate r(T) under P on the market curve (Figure 13),
for the model calibrated without and with the skew, at the paper's parameters.

    python examples/short_rate_density.py [--out out]
"""
import argparse
import pathlib

import jax
import numpy as np
from scipy.stats import gaussian_kde

from slidingcurve import Model, markets, simulate, time_grid, volscale

BP = 1e4
TIMES = (10.0, 20.0, 30.0)
CASES = {"atm": ("ATM vol calibration: sigma = 1.1572%, b = 0", Model(0.03, 5.0, 0.011572)),
         "skew": ("Skew calibration: sigma = 1.1999%, b = 0.367", Model(0.03, 5.0, 0.011999, b=0.367))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = markets.paper_curve("market")
    ref = markets.paper_figures()["fig13"]
    grid = np.asarray(ref["r"])
    t = time_grid(30.0, steps_per_year=12)
    res = {}
    for name, (title, m) in CASES.items():
        ybar = volscale.bootstrap_fn(curve, m, np.arange(1.0, 31.0))(t)
        _, r = simulate(m, curve, t, ybar, jax.random.PRNGKey(1), n_paths=args.paths,
                        snapshot_times=TIMES)
        r = BP * np.asarray(r)
        res[name] = r
        print(f"\n{title}\n    T |   mean (paper) | median | st.dev. | P(r < 0)")
        for i, T in enumerate(TIMES):
            print(f"{T:5.0f} | {r[i].mean():6.0f} ({ref[name]['mean'][f'T={T:.0f}']:6.0f}) | "
                  f"{np.median(r[i]):6.0f} | {r[i].std():7.0f} | {np.mean(r[i] < 0):8.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, (name, (title, _)) in zip(axes, CASES.items()):
        for i, T in enumerate(TIMES):
            sample = res[name][i][:: max(1, res[name].shape[1] // 20000)]
            ax.plot(grid, gaussian_kde(sample)(grid), color=f"C{i}", label=f"T = {T:.0f}y")
            paper = np.array(ref[name]["density"][f"T={T:.0f}"], dtype=float)
            ax.plot(grid[::2], paper[::2], "o", ms=3, color=f"C{i}", markerfacecolor="white")
            ax.axvline(res[name][i].mean(), color=f"C{i}", ls="--", lw=0.8)
        ax.plot([], [], "o", ms=3, color="grey", markerfacecolor="white", label="paper (read off figure)")
        ax.set(xlabel="short rate (bp)", title=title, xlim=(-1500, 2000))
        ax.legend(frameon=False)
    axes[0].set_ylabel("density (1/bp)")
    fig.suptitle("Density of the short rate r(T) under P (cf. Figure 13; dashed: mean)")
    fig.tight_layout()
    fig.savefig(out / "short_rate_density.png", dpi=150)
    print(f"\nfigure written to {out}/short_rate_density.png")


if __name__ == "__main__":
    main()

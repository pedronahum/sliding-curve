"""Figure 2: the yield decomposition on the paper's baseline curve in the
Gaussian case (b = 0, eta = 0, sigma = 1%, kappa = 0.03, delta = 5),
Monte Carlo against the Appendix D closed forms and against the paper's
plotted values (read off Figure 2, see markets.paper_figures).

The skew sweep this example used to carry is superseded by
examples/skew_sweep.py, which calibrates the volatility scaling as the
paper does.

    python examples/gaussian_decomposition.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Model, decompose, gaussian as g, markets, simulate, time_grid, volscale

BP = 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = markets.paper_curve("baseline")
    fig2 = markets.paper_figures()["fig2"]
    model = Model(kappa=0.03, delta=5.0, sigma=0.01)       # Section 9.1 baseline
    t = time_grid(40.0, steps_per_year=12)
    mats = jnp.arange(1.0, 41.0)
    ybar = volscale.ybar_zeroth(curve, model.delta, t)      # immaterial at b = 0

    dec = decompose(simulate(model, curve, t, ybar, jax.random.PRNGKey(0), n_paths=args.paths),
                    model, curve, mats)
    exact = np.array([[g.yield_eh(curve, model.kappa, model.delta, model.sigma, T),
                       g.term_premium(curve, model.kappa, model.delta, model.sigma, T),
                       g.convexity(model.kappa, model.sigma, T)] for T in np.asarray(mats)])
    ft = np.asarray(fig2["T"])
    paper = {k: np.array([fig2[k][int(np.argmin(np.abs(ft - T)))] for T in np.asarray(mats)], dtype=float)
             for k in ("Y", "Y_eh", "theta", "gamma")}
    mc = np.stack([dec.y_eh, dec.theta, dec.gamma], 1)

    print(f"max |Monte Carlo - closed form| = {BP * np.max(np.abs(mc - exact)):.4f}bp; "
          f"max |closed form - Figure 2|: Y_eh {np.max(np.abs(BP*exact[:,0] - paper['Y_eh'])):.2f}, "
          f"theta {np.max(np.abs(BP*exact[:,1] - paper['theta'])):.2f}, "
          f"gamma {np.max(np.abs(BP*exact[:,2] - paper['gamma'])):.2f}bp")
    print(f"{'T':>4} {'Y':>8} {'Y_eh':>8} {'theta':>8} {'gamma':>8} {'resid':>8}   this code (bp)"
          f"  |  Figure 2: {'Y':>7} {'Y_eh':>7} {'theta':>7} {'gamma':>7}")
    for j in (0, 4, 9, 19, 29, 39):
        print(f"{float(dec.T[j]):4.0f} {BP*dec.y[j]:8.2f} {BP*dec.y_eh[j]:8.2f} {BP*dec.theta[j]:8.2f} "
              f"{BP*dec.gamma[j]:8.2f} {BP*dec.residual[j]:8.3f}                 "
              + " ".join(f"{paper[k][j]:7.2f}" for k in ("Y", "Y_eh", "theta", "gamma")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(dec.T, BP * dec.y, color="k", label="Y(0,T) market yield")
    a.plot(dec.T, BP * dec.y_eh, color="C0", label="Y_eh(T) Monte Carlo")
    a.plot(dec.T[::3], BP * exact[::3, 0], "k.", label="closed form")
    a.plot(ft[11::12], np.asarray(fig2["Y_eh"], dtype=float)[11::12], "o", ms=4, color="C0",
           markerfacecolor="white", label="Figure 2 (read off)")
    a.set(xlabel="maturity T (years)", ylabel="basis points", title="Yields")
    a.legend(frameon=False)
    for i, (k, lab) in enumerate((("theta", "theta(T) term premium"), ("gamma", "gamma(T) convexity"))):
        b.plot(dec.T, BP * getattr(dec, k), color=f"C{i + 1}", label=lab)
        b.plot(dec.T[::3], BP * exact[::3, i + 1], "k.", label="closed form" if i == 0 else None)
        b.plot(ft[11::12], np.asarray(fig2[k], dtype=float)[11::12], "o", ms=4, color=f"C{i + 1}",
               markerfacecolor="white", label="Figure 2 (read off)" if i == 0 else None)
    b.set(xlabel="maturity T (years)", ylabel="basis points", title="Term premium and convexity")
    b.legend(frameon=False)
    fig.suptitle("Yield decomposition on the paper's baseline curve, b = 0, eta = 0 (cf. Figure 2)")
    fig.tight_layout()
    fig.savefig(out / "gaussian_decomposition.png", dpi=150)
    print(f"figure written to {out}/gaussian_decomposition.png")


if __name__ == "__main__":
    main()

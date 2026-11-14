"""Figure-2-style yield decomposition in the Gaussian case (b = 0, eta = 0),
Monte Carlo against the Appendix D closed forms, plus an uncalibrated skew
sweep in the style of Figure 3.

The curve is an illustrative Nelson-Siegel curve, NOT the paper's synthetic
curve (which is only shown graphically), so shapes are comparable and levels
are not.

    python examples/gaussian_decomposition.py [--out out]
"""
import argparse
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from slidingcurve import Curve, Model, decompose, gaussian as g, simulate, time_grid, volscale

BP = 1e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--paths", type=int, default=2**16)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    curve = Curve.nelson_siegel(beta0=0.045, beta1=-0.015, beta2=0.01, tau=4.0)
    model = Model(kappa=0.03, delta=5.0, sigma=0.01)       # Section 9.1 baseline
    t = time_grid(40.0, steps_per_year=12)
    mats = jnp.arange(1.0, 41.0)
    key = jax.random.PRNGKey(0)
    ybar = volscale.ybar_zeroth(curve, model.delta, t)

    dec = decompose(simulate(model, curve, t, ybar, key, n_paths=args.paths),
                    model, curve, mats)
    exact = np.array([[g.yield_eh(curve, model.kappa, model.delta, model.sigma, T),
                       g.term_premium(curve, model.kappa, model.delta, model.sigma, T),
                       g.convexity(model.kappa, model.sigma, T)] for T in np.asarray(mats)])

    print(f"{'T':>4} {'Y':>8} {'Y_eh':>8} {'theta':>8} {'gamma':>8} {'resid':>8}   (bp)"
          f"   max |MC - closed form| = "
          f"{BP * np.max(np.abs(np.stack([dec.y_eh, dec.theta, dec.gamma], 1) - exact)):.4f} bp")
    for j in (0, 4, 9, 19, 29, 39):
        print(f"{float(dec.T[j]):4.0f} {BP*dec.y[j]:8.2f} {BP*dec.y_eh[j]:8.2f} "
              f"{BP*dec.theta[j]:8.2f} {BP*dec.gamma[j]:8.2f} {BP*dec.residual[j]:8.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(dec.T, 100 * dec.y, label="Y(0,T) market yield")
    a.plot(dec.T, 100 * dec.y_eh, label="Y_eh(T) Monte Carlo")
    a.plot(dec.T[::3], 100 * exact[::3, 0], "k.", label="closed form")
    a.set(xlabel="maturity T (years)", ylabel="percent", title="Yields")
    a.legend(frameon=False)
    b.plot(dec.T, BP * dec.theta, label="theta(T) term premium")
    b.plot(dec.T, BP * dec.gamma, label="gamma(T) convexity")
    b.plot(dec.T[::3], BP * exact[::3, 1], "k.", label="closed form")
    b.plot(dec.T[::3], BP * exact[::3, 2], "k.")
    b.set(xlabel="maturity T (years)", ylabel="basis points", title="Term premium and convexity")
    b.legend(frameon=False)
    fig.suptitle("Yield decomposition, Gaussian case (b = 0, eta = 0)")
    fig.tight_layout()
    fig.savefig(out / "gaussian_decomposition.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for skew in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = model._replace(b=skew)
        d = decompose(simulate(m, curve, t, ybar, key, n_paths=args.paths), m, curve, mats)
        ax.plot(d.T, BP * d.gamma, label=f"b = {skew}")
    ax.set(xlabel="maturity T (years)", ylabel="basis points",
           title="Convexity vs skew (zeroth-order vol scaling, sigma not recalibrated)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "skew_sweep_uncalibrated.png", dpi=150)
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()

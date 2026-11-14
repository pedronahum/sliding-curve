"""Published test markets.

Only data that could be checked against a primary source is included. So far
that is one synthetic market:

joshi_2014 -- the displaced-diffusion LIBOR market model set-up of Mark
    Joshi's kooderive example (Ex1 in kooexample/main.cpp, the 40-rate
    cancellable swap): semi-annual forwards 0.8% + 0.2% i, displacement 1.5%,
    abcd volatilities a=0.05, b=0.09, c=0.44, d=0.2 and correlation
    exp(-2 * 0.0669 |T_i - T_j|).

    It is a model, not a set of broker quotes. Swaption "quotes" are produced
    from it with the standard frozen-weights (Rebonato) approximation, under
    which the displaced swap rate is lognormal; for one-period swaptions
    (caplets) that is exact. The smile is therefore a pure displaced-diffusion
    skew with no extra curvature.
"""
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from . import bachelier as bl
from .curve import Curve
from .gaussian import _quad
from .pricing import Swaption, annuity, forward_swap_rate


class Quotes(NamedTuple):
    swaption: Swaption
    forward: float              # forward swap rate
    annuity: float
    strikes: np.ndarray
    prices: np.ndarray          # present values of payer swaptions
    normal_vols: np.ndarray     # implied normal volatilities


class DisplacedLMM(NamedTuple):
    curve: Curve
    forwards: np.ndarray        # simple forwards, forwards[i] spans [i tau, (i+1) tau]
    tau: float
    displacement: float
    a: float
    b: float
    c: float
    d: float
    beta: float                 # correlation decay
    L: float = 0.0              # correlation floor

    def _inst_vol(self, reset, t):
        s = reset - t
        return (self.a + self.b * s) * jnp.exp(-self.c * s) + self.d

    def swaption_vol(self, first: int, n: int) -> float:
        """Lognormal volatility of the displaced swap rate over rates
        first .. first + n - 1, expiring at first * tau (frozen weights)."""
        idx = np.arange(first, first + n)
        resets, expiry = idx * self.tau, first * self.tau
        df = np.asarray(self.curve.discount(jnp.asarray(resets + self.tau)))
        w = df / df.sum()
        f = self.forwards[idx] + self.displacement
        s = float(np.sum(w * f))
        var = 0.0
        for i in range(n):
            for j in range(n):
                rho = self.L + (1.0 - self.L) * np.exp(-self.beta * abs(resets[i] - resets[j]))
                cov = float(_quad(lambda t: self._inst_vol(resets[i], t) * self._inst_vol(resets[j], t),
                                  0.0, expiry))
                var += w[i] * w[j] * f[i] * f[j] * rho * cov
        return float(np.sqrt(var / expiry) / s)

    def quotes(self, expiry: float, tenor: float, strike_offsets) -> Quotes:
        first, n = int(round(expiry / self.tau)), int(round(tenor / self.tau))
        swp = Swaption.regular(first * self.tau, n * self.tau, frequency=int(round(1 / self.tau)))
        fwd, ann = float(forward_swap_rate(self.curve, swp)), float(annuity(self.curve, swp))
        strikes = fwd + np.asarray(strike_offsets, dtype=float)
        fwd_prices = np.asarray(bl.displaced_black_price(
            fwd, strikes, self.swaption_vol(first, n), swp.expiry, self.displacement))
        return Quotes(swp, fwd, ann, strikes, ann * fwd_prices,
                      bl.implied_normal_vol(fwd_prices, fwd, strikes, swp.expiry))


def joshi_2014(n_rates: int = 41) -> DisplacedLMM:
    tau = 0.5
    forwards = 0.008 + 0.002 * np.arange(n_rates)
    times = tau * np.arange(n_rates + 1)
    neg_log_df = np.concatenate([[0.0], np.cumsum(np.log1p(tau * forwards))])
    zeros = np.empty_like(times)
    zeros[1:] = neg_log_df[1:] / times[1:]
    zeros[0] = zeros[1]
    # the spline passes through every node, so discount factors on the
    # semi-annual grid, and hence the forwards above, are reproduced exactly
    return DisplacedLMM(Curve.from_zero_nodes(times, zeros), forwards, tau,
                        displacement=0.015, a=0.05, b=0.09, c=0.44, d=0.2, beta=2 * 0.0669)


# --- the paper's synthetic curves, read off its figures -----------------------

_NODES = np.array([0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
                   22.5, 25, 27.5, 30, 32.5, 35, 37.5, 40])


def paper_figures() -> dict:
    """Curves digitised from the paper's Figures 2, 3, 5 and 6 (basis points on
    a quarter-year maturity grid; None where one line hides another). Produced
    by tools/digitise_figures.py; accurate to roughly 0.1-0.3bp."""
    import json
    from importlib.resources import files
    return json.loads(files("slidingcurve").joinpath("data/paper_figures_digitised.json").read_text())


def paper_curve(which: str = "baseline") -> Curve:
    """The paper's synthetic zero curve: 'baseline' (Figure 2) or 'steep'
    (Figure 5), as a spline through yields read off the figure.

    Two assumptions: the yield at T = 0 is extrapolated linearly from the
    first two nodes, and beyond the 40 years the figures show the yield is
    extended quadratically to 45 years (the model needs forwards out to
    T + delta). Forwards amplify digitisation error by a factor of about T,
    so nodes are spaced more widely at the long end.
    """
    fig = paper_figures()[{"baseline": "fig2", "steep": "fig5"}[which]]
    t, y = np.asarray(fig["T"]), np.asarray(fig["Y"], dtype=float) * 1e-4
    nodes = np.interp(_NODES, t, y)
    y0 = nodes[0] - (nodes[1] - nodes[0]) / (_NODES[1] - _NODES[0]) * _NODES[0]
    quad = np.polyfit(_NODES[-4:], nodes[-4:], 2)
    ext = np.array([42.5, 45.0])
    return Curve.from_zero_nodes(np.concatenate([[0.0], _NODES, ext]),
                                 np.concatenate([[y0], nodes, np.polyval(quad, ext)]))

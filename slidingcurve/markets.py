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


_MARKET_NODES = np.array([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10,
                          12, 14, 16, 18, 20, 22.5, 25, 27.5, 30])


def paper_smile():
    """Section 9.4 smile on the 20y x 5y yield option, read off Figure 11:
    strike offsets from at-the-money, the market's normal vols and the vols
    of the paper's own fit (all in decimal units)."""
    f = paper_figures()["fig11"]
    k = np.asarray(f["strikes"])
    return (k - k[2]) * 1e-4, np.asarray(f["market"]) * 1e-4, np.asarray(f["paper_fit"]) * 1e-4


def paper_curve(which: str = "baseline") -> Curve:
    """A zero curve from the paper, as a spline through yields read off a
    figure: the synthetic 'baseline' (Figure 2) and 'steep' (Figure 5) curves,
    or the 21-Aug-2026 USD 'market' curve of Section 9.4 (Figure 10).

    Two assumptions: the yield at T = 0 is extrapolated linearly from the
    first two nodes, and beyond the last maturity the figures show (40 years,
    30 for the market curve) the yield is extended quadratically by five years
    (the model needs forwards out to T + delta). Forwards amplify digitisation error by a factor of about T,
    so nodes are spaced more widely at the long end.
    """
    fig = paper_figures()[{"baseline": "fig2", "steep": "fig5", "market": "fig10"}[which]]
    _NODES = _MARKET_NODES if which == "market" else globals()["_NODES"]
    t, y = np.asarray(fig["T"]), np.asarray(fig["Y"], dtype=float) * 1e-4
    nodes = np.interp(_NODES, t, y)
    y0 = nodes[0] - (nodes[1] - nodes[0]) / (_NODES[1] - _NODES[0]) * _NODES[0]
    quad = np.polyfit(_NODES[-4:], nodes[-4:], 2)
    ext = _NODES[-1] + np.array([2.5, 5.0])
    return Curve.from_zero_nodes(np.concatenate([[0.0], _NODES, ext]),
                                 np.concatenate([[y0], nodes, np.polyval(quad, ext)]))


# --- a second smile market from a published source ------------------------------

CHOI_2017_SOURCE = (
    "J. Choi, C. Liu, B. K. Seo, 'Hyperbolic normal stochastic volatility model', "
    "Journal of Futures Markets 39(2), 2019; arXiv:1809.04035, Section 4.1 and Tables 3-4. "
    "USD swaptions, Reuters, 14 March 2017.")


def choi_2017_10y10y():
    """Three market prices of the USD 10y x 10y swaption smile, 14 March 2017.

    The paper does not tabulate its market quotes (they appear only in its
    Figure 3); its Table 4 gives prices of models that Section 4.1 says were
    calibrated to reproduce the market at strikes F, F +- 1% exactly. At
    those three strikes the two calibrated models (lambda = 0 and 1) print
    identical prices to four figures, as exact calibration requires; those
    are the values here. Prices are in units of the annuity; the forward
    swap rate is F = 3.0673% (Table 3); expiry 10 years.

    What the source does not give: the discount curve and the swap's fixed-
    leg conventions. Anything that needs them (a pricer in this model) must
    assume them; see examples/second_market.py.

    Returns (expiry, forward, strike offsets, prices per unit annuity,
    implied normal vols), decimals.
    """
    expiry, fwd = 10.0, 0.030673
    offsets = np.array([-0.01, 0.0, 0.01])
    prices = np.array([1.506e-2, 9.083e-3, 5.108e-3])     # Table 4, K - F = -100, 0, +100bp
    vols = bl.implied_normal_vol(prices, fwd, fwd + offsets, expiry)
    return expiry, fwd, offsets, prices, vols

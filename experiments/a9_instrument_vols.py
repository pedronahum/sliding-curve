"""A9/A4: exact Gaussian (b = 0, eta = 0) normal vols of candidate 'yield
option' instruments on the 20y x 5y benchmark, at sigma = 1%, and the sigma
each needs for an 82bp at-the-money vol on the market curve (paper: 1.157%).

Candidates: the continuously compounded forward yield Y1 (any measure: its
variance is measure-independent at b = 0), the simple rate L(T, T+5)
(caplet, measure of T+5), and par swap rates with annual / semi-annual /
quarterly fixed legs (annuity measure; Jamshidian exact)."""
import numpy as np
from slidingcurve import markets, gaussian as g, bachelier as bl, pricing as pr

BP = 1e4
kappa, delta, T = 0.03, 5.0, 20.0


def swaption_vol(curve, sigma, freq):
    swp = pr.Swaption.regular(T, delta, freq) if freq else pr.Swaption.caplet(T, delta)
    F, ann = float(pr.forward_swap_rate(curve, swp)), float(pr.annuity(curve, swp))
    p = g.payer_swaption(curve, kappa, sigma, T, np.asarray(swp.pay_times), np.asarray(swp.accruals), F)
    return float(bl.implied_normal_vol(p / ann, F, F, T))


for name in ("baseline", "market"):
    curve = markets.paper_curve(name)
    print(f"\n{name} curve, sigma = 1%:")
    yv = g.A(kappa, delta) * 0.01 * np.sqrt(float(g.B(2 * kappa, T)) / T)
    rows = [("continuous yield Y1", yv)] + [(lab, swaption_vol(curve, 0.01, f)) for lab, f in
            (("simple rate L(T,T+5) (caplet)", None), ("par swap, annual", 1),
             ("par swap, semi-annual", 2), ("par swap, quarterly", 4))]
    for lab, v in rows:
        print(f"  {lab:32s} {BP*v:7.2f}bp   sigma for 82bp: {0.82/ (BP*v) * 100:.4f}%")

"""A9/A1: which quantity is the paper's 'ATM' strike in Figures 4 and 8?

Figure 4 marks ATM at 370, 390, 395.5, 387.5bp (5y, 10y, 15y, 20y expiry) and
Figure 8 at 387bp (20y). Compare with candidate forwards on the baseline curve.
"""
import numpy as np
from slidingcurve import markets, gaussian as g

BP = 1e4
curve = markets.paper_curve("baseline")
kappa, delta, sigma = 0.03, 5.0, 0.01
paper_atm = {5: 370.0, 10: 390.0, 15: 395.5, 20: 387.5}
print(f"Y1(0) = Y(0,0,5) = {BP*float(curve.fwd_yield(0.0, delta)):.2f}bp")
print("  T | paper ATM | Y(0,T,T+5) | simple L(0,T,T+5) | E^{T+d}[Y1] | E^T[Y1] | E^Q(spot)[Y1]")
for T, atm in paper_atm.items():
    fy = float(curve.fwd_yield(T, T + delta))
    L = (float(curve.discount(T)) / float(curve.discount(T + delta)) - 1) / delta
    Ad, Bd = g.A(kappa, delta), g.B(kappa, delta)
    y = float(g.y_det(kappa, sigma, T))
    m_Td = float(g.yhat(curve, kappa, delta, sigma, T, T))         # measure of T+delta
    m_T = fy + Ad * float(g.mean_x_fwd(kappa, sigma, T, T)) + Bd**2 / (2 * delta) * y
    # spot Q: E^Q[x(T)] = int_0^T e^{-k(T-s)} y(s) ds
    s = np.linspace(0, T, 4001)
    yq = sigma**2 * (1 - np.exp(-2 * kappa * s)) / (2 * kappa)
    ex = np.trapezoid(np.exp(-kappa * (T - s)) * yq, s)
    m_Q = fy + Ad * ex + Bd**2 / (2 * delta) * y
    print(f"{T:3d} | {atm:9.1f} | {BP*fy:10.2f} | {BP*L:17.2f} | {BP*m_Td:11.2f} | {BP*m_T:7.2f} | {BP*m_Q:8.2f}")

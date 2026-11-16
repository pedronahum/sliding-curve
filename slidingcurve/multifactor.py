"""Multi-factor quasi-Gaussian model in the sliding-curve measure (Sections
5-6, Appendix C), with constant mean reversions.

d factors, h_i(t) = exp(-kappa_i t), benchmark tenors
0 = delta_0 < delta_1 < ... < delta_d and benchmark forward yields
Y_i(t) = Y(t, t + delta_{i-1}, t + delta_i). With constant kappa the matrices
of Section 6 simplify:

    H^Y(t) = M H(t),     M_ij = exp(-kappa_j delta_{i-1}) A(kappa_j, Delta_i),
    H(t) H^Y(t)^-1 = M^-1                               (constant)
    Y(t) = Y_f(0, t) + M x + Gamma(y) / 2,                                   (44)
    Gamma_i(y) = (g_i' y g_i - g_{i-1}' y g_{i-1}) / Delta_i,   g_i = B(kappa, delta_i)
    sigma_r' sigma_r = z M^-1 S X S M^-T,   S = diag(sigma^Y_i)              (56)
    dx = (y 1 - kappa x - M^-1 mu^Y) dt + sigma_r' dW^P                      (52)
    mu^Y_i = rho_i + sum_kl W^(i)_kl (sigma_r' sigma_r)_kl                   (27), (45)
    W^(i)_kl = (1/Delta_i) int_{delta_{i-1}}^{delta_i} exp(-kappa_k s) B(kappa_l, s) ds

The local volatility generalises the basis-point convention of Section 9.1:
sigma^Y_i = s_i ((1 - b_i) + b_i Y_i / Ybar_i(t)) sqrt(z), floored at zero
and capped at vol_cap * s_i; s_i is the basis-point vol of benchmark i. With
d = 1 and s_1 = A(delta) sigma this is the one-factor model of model.py.

Discretisation as in model.py: y exact for frozen volatility; the drift of x
is the increment that keeps every benchmark yield an exact discrete P-
martingale, M E[dx] = -(dY_f(0, t) + dGamma / 2).
"""
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from ._typing import Array, ArrayLike, FloatSeq
from .curve import Curve
from .gaussian import A as _A, B as _B, _quad
from .model import variance_step


class MultiModel(NamedTuple):
    kappa: Sequence[float]      # mean reversions, length d
    deltas: Sequence[float]     # benchmark tenors delta_1 < ... < delta_d (delta_0 = 0)
    vols: Sequence[float]       # basis-point vols s_i of the benchmark yields
    corr: ArrayLike | FloatSeq  # d x d correlation X^Y of the benchmark yields
    skews: Sequence[float] | None = None
    eta: float = 0.0
    kappa_z: float = 0.1
    vol_cap: float = 10.0
    variance_scheme: str = "qe"

    @property
    def d(self) -> int:
        return len(self.kappa)

    def arrays(self) -> tuple[Array, Array, Array, Array, Array]:
        k = jnp.asarray(self.kappa, dtype=float)
        dl = jnp.concatenate([jnp.zeros(1), jnp.asarray(self.deltas, dtype=float)])
        b = jnp.zeros(self.d) if self.skews is None else jnp.asarray(self.skews, dtype=float)
        return k, dl, jnp.asarray(self.vols, dtype=float), jnp.asarray(self.corr, dtype=float), b


def _Bv(k: ArrayLike, s: ArrayLike) -> Array:
    return _B(k, s)


def structure(m: MultiModel) -> tuple[Array, Array, Array, Array, Array]:
    """Constant matrices: M, M^-1, g_i = B(kappa, delta_i) (rows, i = 0..d),
    Delta_i and the convexity weights W^(i)."""
    k, dl, _, _, _ = m.arrays()
    D = dl[1:] - dl[:-1]
    M = jnp.exp(-k[None, :] * dl[:-1, None]) * _A(k[None, :], D[:, None])
    g = _Bv(k[None, :], dl[:, None])                                   # [d+1, d]

    def w_entry(kk: Array, kl: Array, a: float | Array, b_: float | Array) -> Array:
        return _quad(lambda s: jnp.exp(-kk * s) * _B(kl, s), a, b_)

    W = jnp.stack([jnp.stack([jnp.stack([w_entry(k[p], k[q], dl[i], dl[i + 1]) for q in range(m.d)])
                              for p in range(m.d)]) / D[i] for i in range(m.d)])   # [d, d, d]
    return M, jnp.linalg.inv(M), g, D, W


def gamma_y(y: Array, g: Array, D: Array) -> Array:
    """Gamma_i(y) of (44), for y of shape [..., d, d]."""
    q = jnp.einsum("id,...de,ie->...i", g, y, g)
    return (q[..., 1:] - q[..., :-1]) / D


def fwd_benchmarks(curve: Curve, dl: Array, t: ArrayLike) -> Array:
    """Y_f(0, t) = (Y(0, t + delta_{i-1}, t + delta_i))_i on times t: [len(t), d]."""
    t = jnp.asarray(t, dtype=float)[:, None]
    return curve.fwd_yield(t + dl[None, :-1], t + dl[None, 1:])


class MultiProfiles(NamedTuple):
    t: jax.Array
    x: jax.Array            # [K, N+1, d]   E[x]
    nu: jax.Array           # [K, N+1, d]   E[M^-1 mu^Y]  (the term-premium integrand, (54))
    V: jax.Array            # [K, N+1, d, d] E[sigma_r' sigma_r]
    Y: jax.Array            # [K, N+1, d]   E[Y(t)] (P-martingales)


def simulate_multi(m: MultiModel, curve: Curve, t: Array, ybar: ArrayLike | FloatSeq, key: Array,
                   n_paths: int = 2**14, n_batches: int = 16) -> MultiProfiles:
    """Simulate (52) on the uniform grid t. `ybar` is [len(t), d]: the
    volatility scaling of each benchmark."""
    k, dl, s, X, b = m.arrays()
    d = m.d
    M, Minv, g, D, W = structure(m)
    Dchol = jnp.linalg.cholesky(X)                     # X = L L', so D^Y = L'
    dt = t[1] - t[0]
    sq = jnp.sqrt(dt)
    kk = k[:, None] + k[None, :]
    decay, gain = jnp.exp(-kk * dt), _Bv(kk, dt)
    Yf = fwd_benchmarks(curve, dl, t)
    dYf = fwd_benchmarks(curve, dl, t + dt) - Yf
    per = n_paths // n_batches
    half = per // 2
    ybar = jnp.asarray(ybar, dtype=float)

    def fwd_at(T: ArrayLike) -> Array:                 # f(0, T), broadcast
        return curve.forward(T)

    tau_ends = jnp.stack([t[:, None] + dl[None, :-1], t[:, None] + dl[None, 1:]], 0)   # [2, N+1, d]
    f_lo, f_hi = fwd_at(tau_ends[0]), fwd_at(tau_ends[1])
    xs = (jnp.arange(t.shape[0]), Yf, dYf, ybar, f_lo, f_hi)

    def step(state: tuple[Array, Array, Array],
             inp: tuple[Array, Array, Array, Array, Array, Array],
             ) -> tuple[tuple[Array, Array, Array], tuple[Array, Array, Array, Array]]:
        x, y, z = state                                 # [K, P, d], [K, P, d, d], [K, P]
        i, yf, dyf, yb, flo, fhi = inp
        Y = yf + jnp.einsum("ij,kpj->kpi", M, x) + 0.5 * gamma_y(y, g, D)
        lv = s * jnp.maximum((1.0 - b) + b * Y / yb, 0.0)
        sv = jnp.minimum(jnp.sqrt(jnp.maximum(z, 1e-12))[..., None] * lv, m.vol_cap * s)   # sigma^Y_i
        # sigma_r' = M^-1 S L ; V = sigma_r' sigma_r = M^-1 S X S M^-T
        SL = sv[..., :, None] * Dchol[None, None]
        sr = jnp.einsum("ij,kpjl->kpil", Minv, SL)                                  # sigma_r'
        V = jnp.einsum("kpil,kpjl->kpij", sr, sr)
        # roll: f(t, T) = f(0, T) + sum_j e^{-kappa_j (T - t)} (x_j + (y G(t, T))_j)
        def f_t(f0: Array, off: Array) -> Array:
            G = _Bv(k[None, :], off[:, None])                                       # [d(window), d]
            yG = jnp.einsum("kpjl,wl->kpwj", y, G)
            e = jnp.exp(-k[None, :] * off[:, None])
            return f0 + jnp.einsum("wj,kpwj->kpw", e, x[:, :, None, :] + yG)
        rho = (f_t(fhi, dl[1:]) - f_t(flo, dl[:-1])) / D
        mu = rho + jnp.einsum("iab,qpab->qpi", W, V)
        nu = jnp.einsum("ij,kpj->kpi", Minv, mu)
        stats = (x.mean(1), nu.mean(1), V.mean(1), Y.mean(1))
        xi = jax.random.normal(jax.random.fold_in(key, i), (d + 1, n_batches, half))
        xi = jnp.concatenate([xi, -xi], axis=2)
        y_new = y * decay + V * gain
        dG = gamma_y(y_new, g, D) - gamma_y(y, g, D)
        drift = -jnp.einsum("ij,kpj->kpi", Minv, dyf + 0.5 * dG)
        x_new = x + drift + sq * jnp.einsum("kpij,jkp->kpi", sr, xi[:d])
        z_new = variance_step(m, z, xi[d], dt)
        return (x_new, y_new, z_new), stats

    init = (jnp.zeros((n_batches, per, d)), jnp.zeros((n_batches, per, d, d)), jnp.ones((n_batches, per)))
    _, (ex, enu, eV, eY) = jax.lax.scan(step, init, xs)
    sw: Callable[[Array], Array] = lambda a: jnp.swapaxes(a, 0, 1)
    return MultiProfiles(t, sw(ex), sw(enu), sw(eV), sw(eY))


class MultiDecomposition(NamedTuple):
    T: jax.Array
    y: jax.Array
    y_eh: jax.Array
    theta: jax.Array
    gamma: jax.Array
    residual: jax.Array
    y_eh_se: jax.Array
    theta_se: jax.Array
    gamma_se: jax.Array
    residual_se: jax.Array


def decompose_multi(p: MultiProfiles, m: MultiModel, curve: Curve,
                    maturities: ArrayLike | FloatSeq) -> MultiDecomposition:
    """(13), (54), (55) as quadratures of the profiles (G(t, T) deterministic)."""
    k = jnp.asarray(m.kappa, dtype=float)
    t = p.t
    dt = t[1] - t[0]
    T = jnp.asarray(maturities, dtype=float)
    n = jnp.round(T / dt).astype(int)
    T = t[n]
    i = jnp.arange(t.shape[0])
    w = dt * jnp.where(i[None, :] < n[:, None], 1.0, 0.0)
    w = w.at[:, 0].mul(0.5) + 0.5 * dt * (i[None, :] == n[:, None])          # [Mt, N+1]
    G = _Bv(k[None, None, :], jnp.maximum(T[:, None, None] - t[None, :, None], 0.0))   # [Mt, N+1, d]
    y = curve.zero(T)
    y_eh = y + jnp.einsum("mn,knd->km", w, p.x) / T
    theta = jnp.einsum("mn,mnd,knd->km", w, G, p.nu) / T
    gamma = jnp.einsum("mn,mnd,kndf,mnf->km", w, G, p.V, G) / (2.0 * T)
    res = y_eh + theta - gamma - y
    kb = p.x.shape[0]
    mean: Callable[[Array], Array] = lambda a: jnp.mean(a, 0)
    se: Callable[[Array], Array] = lambda a: jnp.std(a, 0, ddof=1) / jnp.sqrt(kb)
    return MultiDecomposition(T, y, mean(y_eh), mean(theta), mean(gamma), mean(res),
                              se(y_eh), se(theta), se(gamma), se(res))


# --- Gaussian closed forms (b = 0, eta = 0) -------------------------------------

def gaussian_V(m: MultiModel) -> Array:
    """Constant sigma_r' sigma_r = M^-1 S X S M^-T with S = diag(s)."""
    _, _, s, X, _ = m.arrays()
    _, Minv, _, _, _ = structure(m)
    return Minv @ (s[:, None] * X * s[None, :]) @ Minv.T


def gaussian_y(m: MultiModel, t: ArrayLike) -> Array:
    """y(t)_ij = V_ij B(kappa_i + kappa_j, t)."""
    k = jnp.asarray(m.kappa, dtype=float)
    return gaussian_V(m) * _Bv(k[:, None] + k[None, :], jnp.asarray(t, dtype=float))


def gaussian_mean_x_P(m: MultiModel, curve: Curve, t: ArrayLike) -> Array:
    """E^P[x(t)] = M^-1 (Y(0) - Y_f(0, t) - Gamma(y(t)) / 2): the martingale
    condition (26) with a deterministic y."""
    _, dl, _, _, _ = m.arrays()
    M, Minv, g, D, _ = structure(m)
    Y0 = fwd_benchmarks(curve, dl, jnp.zeros(1))[0]
    Yf = fwd_benchmarks(curve, dl, jnp.atleast_1d(t))
    yt = jax.vmap(lambda s: gaussian_y(m, s))(jnp.atleast_1d(t))
    return jnp.einsum("ij,nj->ni", Minv, Y0[None] - Yf - 0.5 * gamma_y(yt, g, D))


def gaussian_mean_x_fwd(m: MultiModel, t: ArrayLike, tau: ArrayLike) -> Array:
    """E^{Q^tau}[x(t)] = -y(t) G(t, tau), generalising (90)."""
    k = jnp.asarray(m.kappa, dtype=float)
    return -gaussian_y(m, t) @ _Bv(k, jnp.asarray(tau) - jnp.asarray(t))


def gaussian_decomposition(m: MultiModel, curve: Curve, T: float) -> tuple[Array, Array, Array]:
    """(Y_eh, theta, gamma) at maturity T, Gaussian case, by quadrature."""
    k = jnp.asarray(m.kappa, dtype=float)
    V = gaussian_V(m)
    gam = _quad(lambda t: jax.vmap(lambda s: _Bv(k, T - s) @ V @ _Bv(k, T - s))(t), 0.0, T) / (2.0 * T)
    yeh = curve.zero(T) + _quad(lambda t: gaussian_mean_x_P(m, curve, t).sum(1), 0.0, T) / T
    return yeh, curve.zero(T) + gam - yeh, gam


# --- volatility scaling: zeroth order (79) and the general bootstrap (C.3.2) -----

def ybar_zeroth(m: MultiModel, curve: Curve, t: ArrayLike) -> Array:
    """Ybar_j(t) = Y(0, t + delta_{j-1}, t + delta_j), eq. (79)."""
    _, dl, _, _, _ = m.arrays()
    return fwd_benchmarks(curve, dl, t)


def yhat_multi(m: MultiModel, curve: Curve, t: ArrayLike, tau: ArrayLike) -> Array:
    """E^{Q^tau}[Y_j(t)] in the Gaussian model: [d]."""
    _, dl, _, _, _ = m.arrays()
    M, _, g, D, _ = structure(m)
    return (fwd_benchmarks(curve, dl, jnp.atleast_1d(t))[0] + M @ gaussian_mean_x_fwd(m, t, tau)
            + 0.5 * gamma_y(gaussian_y(m, t), g, D))


def ybar_bootstrap_multi(m: MultiModel, curve: Curve, expiries: ArrayLike | FloatSeq,
                         weighted: bool = False, zeroth_mean: bool = False) -> Array:
    """General bootstrap of Appendix C.3.2 with piecewise-constant Ybar_j on
    (T_{m-1}, T_m]. For each caplet tenor i (the rate L_i on [T_m, T_m + delta_i])
    and expiry T_m,

        sum_j int_0^{T_m} nabla_ji(t, T_m) (1 - b_j) (Ybar_j(t) - Yhat_j^(i)(t, T_m)) dt = 0,

    with Yhat_j^(i)(t, T_m) = E^{Q^{T_m + delta_i}}[Y_j(t)] in closed form (Gaussian)
    rather than approximation (78), and the deterministic Jacobian (77)
    nabla_.i(t, T_m) = J_i(t, T_m) M^-1, J_i = dY(0, T_m, T_m + delta_i)/dx(t),
    approximating L_i by the yield as in C.3.3. weighted=False freezes it at
    t = T_m (the variant of Section 9.1: in one factor, nabla = 1);
    weighted=True keeps the t-dependence (exp(kappa t) weights of (80)).
    Integrals use the rectangle rule of (81) (right end points), so d = 1
    reproduces volscale.ybar_bootstrap exactly. zeroth_mean=True uses (78)
    instead of the closed form; the solution is then (79), the forwards.

    Returns [len(expiries), d]."""
    k, dl, _, _, b = m.arrays()
    d = m.d
    _, Minv, _, _, _ = structure(m)
    Tm = np.asarray(expiries, dtype=float)
    lo = np.concatenate([[0.0], Tm[:-1]])
    K = Tm - lo
    out: list[Array] = []
    for mi in range(len(Tm)):
        T = Tm[mi]

        def nabla(t: Any, i: int) -> Array:                # row over j
            tt = T if not weighted else t
            J = jnp.exp(-k * (T - tt)) * _A(k, dl[i + 1])
            return J @ Minv

        def yhat(t: Any, i: int) -> Array:
            if zeroth_mean:
                return fwd_benchmarks(curve, dl, jnp.atleast_1d(t))[0]
            return yhat_multi(m, curve, t, T + dl[i + 1])

        C = jnp.stack([nabla(T, i) * (1.0 - b) * K[mi] for i in range(d)])            # [i, j]
        rhs_list: list[Array] = []
        for i in range(d):
            acc = sum(jnp.dot(nabla(Tm[q], i) * (1.0 - b), yhat(Tm[q], i)) * K[q] for q in range(mi + 1))
            acc = acc - sum(jnp.dot(nabla(Tm[q], i) * (1.0 - b), out[q]) * K[q] for q in range(mi))
            rhs_list.append(acc)
        rhs = jnp.stack(rhs_list)
        active = (1.0 - b) != 0.0
        # b_j = 1: the equation does not involve Ybar_j, which drops out of the model
        Ca = jnp.where(active[None, :], C, jnp.eye(d))
        ra = jnp.where(active, rhs, 0.0) + jnp.where(active, 0.0,
                                                     fwd_benchmarks(curve, dl, jnp.atleast_1d(T))[0])
        out.append(jnp.linalg.solve(Ca, ra))
    return jnp.stack(out)


def piecewise(expiries: ArrayLike | FloatSeq, values: ArrayLike | FloatSeq,
              t: ArrayLike) -> Array:
    Tm = jnp.asarray(expiries, dtype=float)
    idx = jnp.clip(jnp.searchsorted(Tm, jnp.asarray(t, dtype=float), side="left"), 0, Tm.shape[0] - 1)
    return jnp.asarray(values)[idx]

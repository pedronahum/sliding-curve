"""B1: a pathwise-differentiable variance scheme.

Compares the library's QE scheme, the moment-matched lognormal scheme and
exact CIR sampling (noncentral chi-square, numpy) on
 (a) moments of z(T) and of int z,
 (b) the 20y x 5y yield-option smile at eta = 1.5, b = 0 (Figure 8 set-up)
     and at the paper's three-parameter fit (sigma 1.492%, b 0.417,
     eta 0.779, market curve),
 (c) d(price)/d(eta): autodiff through the lognormal scheme against
     central finite differences with common random numbers.
"""
import numpy as np
import jax
import jax.numpy as jnp
from common import BP, smile, save, exact_cir, scheme_paths, yield_from_z
from slidingcurve import Model, bachelier as bl, markets, pricing as pr, volscale
from slidingcurve.gaussian import A, B
from slidingcurve.model import local_vol, variance_step

N = 2**15
T = 20.0
out = {}
# (a) moments
eta, kz = 1.5, 0.1
var_exact = eta**2 / (2 * kz) * (1 - np.exp(-2 * kz * T))
print(f"(a) eta {eta}: exact E[z(T)] = 1, Var[z(T)] = {var_exact:.3f}")
zs = {"exact": exact_cir(eta, kz, N, 0)}
for sch in ("qe", "lognormal"):
    zs[sch] = scheme_paths(Model(0.03, 5.0, 0.01, eta=eta, kappa_z=kz, variance_scheme=sch), N, jax.random.PRNGKey(7))
for k, z in zs.items():
    I = z[1:].mean(axis=0) * T
    out[f"moments|{k}"] = dict(mean=float(z[-1].mean()), var=float(z[-1].var()), int_mean=float(I.mean()),
                               int_var=float(I.var()), p_tiny=float(np.mean(z[-1] < 1e-3)))
    print(f"   {k:9s} E[z(T)] {z[-1].mean():.3f} +- {z[-1].std()/np.sqrt(N):.3f}  Var {z[-1].var():6.3f}  "
          f"E[int z] {I.mean():6.2f}  sd[int z] {I.std():6.2f}  P(z(T) < 1e-3) {np.mean(z[-1] < 1e-3):.3f}")

# (b) smiles
cases = {"fig8 eta=1.5": (Model(0.03, 5.0, 0.01, eta=1.5), "baseline", None),
         "3-param fit": (Model(0.03, 5.0, 0.01492, b=0.417, eta=0.779), "market", "boot")}
off = 1e-4 * np.array([-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0])
for name, (m, cname, sc) in cases.items():
    curve = markets.paper_curve(cname)
    fn = (volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0)) if sc else
          (lambda t: volscale.ybar_zeroth(curve, 5.0, t)))
    print(f"(b) {name}: smile at mean + {np.round(BP*off).astype(int)} bp")
    vols = {}
    for k in ("exact", "qe", "lognormal"):
        z = exact_cir(m.eta, m.kappa_z, N, 1) if k == "exact" else scheme_paths(
            m._replace(variance_scheme=k), N, jax.random.PRNGKey(11))
        y1 = yield_from_z(m, curve, fn, z, jax.random.PRNGKey(3))
        vols[k] = BP * smile(y1, y1.mean() + off, y1.mean(), T)
        # standard error of the ATM vol by batches
        bat = [BP * smile(b_, b_.mean() + off[3:4], b_.mean(), T)[0] for b_ in np.array_split(y1, 16)]
        out[f"smile|{name}|{k}"] = dict(vols=vols[k].tolist(), atm_se=float(np.std(bat, ddof=1) / 4))
        print(f"   {k:9s} {np.round(vols[k], 2)}  (ATM s.e. {np.std(bat, ddof=1)/4:.2f})")

# (c) derivative in eta, lognormal scheme, autodiff vs finite differences
curve = markets.paper_curve("market")
m = Model(0.03, 5.0, 0.01492, b=0.417, eta=0.779, variance_scheme="lognormal")
fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
offs = jnp.array([-0.02, 0.0, 0.02])
price = lambda e: pr.yield_option_prices(m._replace(eta=e), curve, T, offs, fn, jax.random.PRNGKey(5), 2**14)[1]
g_ad = np.asarray(jax.jacfwd(price)(0.779))
fd = {h: (np.asarray(price(0.779 + h)) - np.asarray(price(0.779 - h))) / (2 * h) for h in (1e-2, 1e-3, 1e-4)}
print("(c) d price / d eta, lognormal scheme: autodiff", g_ad, " central FD", {h: v for h, v in fd.items()})
mq = m._replace(variance_scheme="qe")
price_q = lambda e: pr.yield_option_prices(mq._replace(eta=e), curve, T, offs, fn, jax.random.PRNGKey(5), 2**14)[1]
fdq = {h: (np.asarray(price_q(0.779 + h)) - np.asarray(price_q(0.779 - h))) / (2 * h) for h in (1e-2, 1e-3, 1e-4)}
g_adq = np.asarray(jax.jacfwd(price_q)(0.779))
print("    QE scheme: autodiff", g_adq, " central FD", fdq)
out["deriv"] = dict(ln_ad=g_ad.tolist(), ln_fd={str(h): v.tolist() for h, v in fd.items()},
                    qe_ad=g_adq.tolist(), qe_fd={str(h): v.tolist() for h, v in fdq.items()})
save("b1_variance_schemes", out)

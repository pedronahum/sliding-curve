"""A1: search for a coding error that reproduces Figure 4 (PRE-REGISTERED).

This docstring, the hypothesis list and the decision rules were committed
before the script was first run. Hypotheses added after seeing results must
be marked POST HOC in the log and in docs/investigation.md.

Question. Figure 4 disagrees with the paper's own text and its Figure 8 (see
docs/investigation.md, A1). Is there a single, natural coding error, with no
free parameter, that turns the model's correct smiles into the published
figure? Only a hypothesis that reproduces the figure would be reported, and
then only as "consistent with"; the code is the authors'.

Fixed setup (as in a1_conventions.py): baseline curve, kappa = 0.03,
delta = 5, sigma = 1%, eta = 0, b in {0, 0.25, 0.5, 0.75, 1}, bootstrap
volatility scaling on an annual grid, expiries 5, 10, 15, 20y, 24 steps a
year, 2^17 antithetic paths, seed PRNGKey(4).

Hypotheses (every one has zero free parameters):

  P-family: a pricing recipe = distribution x inversion forward x option
  rule x strike centre.
    distribution  T+d       Y_1(T) in Q^(T+delta) (the library)
                  T         Y_1(T) in Q^T (= spot Q, pathwise discounted)
                  Q-nodisc  Y_1(T) in spot Q, discount factor forgotten
                  P         Y_1(T) in P, undiscounted
                  P-disc    Y_1(T) in P, weighted by the pathwise discount
    inversion     mean      the sample mean of that distribution
    forward       curve     Y(0, T, T+delta)
                  spot      Y(0, 0, delta) (the sliding benchmark today)
    option rule   call | put | otm (puts below the inversion forward) |
                  itm (calls below, puts above: the otm test with its
                  sign flipped)
    strikes       curve     curve forward + offsets (the paper's axis)
                  mean      distribution mean + offsets
  With the mean as inversion forward the option rule is irrelevant (exact
  in-sample parity); those collapse to one hypothesis each.

  N-family: a numeraire left in the price; library distribution (T+d),
  otm, strikes from the curve forward, inversion at the mean or the curve:
    N1  price x P(0, T+delta) / P(0, T)  (payment discount not undone)
    N2  price x P(0, T)                  (deflated price not divided)

  D-family: an error in the local volatility; library pricing (T+d, mean,
  strikes from the mean):
    D1  local variance linear in Y: sigma * sqrt((1 - b) + b Y / Ybar)
    D2  local vol squared:          sigma * ((1 - b) + b Y / Ybar)^2

Scoring. Mean absolute error (bp) per panel over points both present.
Signatures on the 20y panel, each pass/fail:
  S1  b = 0 droops: vol at +120 at least 8bp below its ATM, and +160 and
      +200 absent or at least 20bp below ATM (paper: -13.3, absent)
  S2  ATM rises with skew: ATM(b = 1) - ATM(b = 0) > +2bp (paper: +5.9)
  S3  steep skew: b = 1 vol(+200) - vol(-80) >= 80bp (paper: 109)
  S4  b = 1 at -200, -160, -120 all absent (price at or below intrinsic)

Decision rules (selection on 20y, validation on the held-out 5, 10, 15y):
  reproduces    S1-S4 hold, 20y error <= 3bp, and each held-out panel
                <= 3bp
  shape only    S1-S4 hold, errors larger: reported, not an explanation
  refuted       otherwise
The 3bp bound is about three times the panel errors the library reaches
where it matches (5y: 1.1bp) and well above digitisation error (~0.3bp).
With ~100 zero-parameter hypotheses and 220 points, passing by chance is
not a concern; failing everything is the expected outcome and says only
that the error, if there is one, is not on this list.

Usage: XLA_PYTHON_CLIENT_PREALLOCATE=false ../.venv-gpu/bin/python a1_bug_search.py
"""
import itertools
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp

from common import BP, terminal_yield, implied, save, OUT, use_local_vol
from slidingcurve import Model, markets, volscale

jax.config.update("jax_enable_x64", True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2**17
KEY = jax.random.PRNGKey(4)
EXPIRIES = (5, 10, 15, 20)
SKEWS = (0.0, 0.25, 0.5, 0.75, 1.0)
DELTA = 5.0
curve = markets.paper_curve("baseline")
fig4 = markets.paper_figures()["fig4"]
base = Model(kappa=0.03, delta=DELTA, sigma=0.01)
grid = np.arange(1.0, 21.0)
OFFS = np.asarray(fig4["20"]["offsets"]) * 1e-4
I = {o: i for i, o in enumerate(fig4["20"]["offsets"])}


# --- local-volatility errors (D-family) ----------------------------------------

def lv_variance_linear(model, y1, ybar, z):
    v = jnp.maximum((1.0 - model.b) + model.b * y1 / ybar, 0.0)
    return jnp.minimum(jnp.sqrt(jnp.maximum(z, 1e-12)) * model.sigma * jnp.sqrt(v),
                       model.vol_cap * model.sigma)


def lv_squared(model, y1, ybar, z):
    v = jnp.maximum((1.0 - model.b) + model.b * y1 / ybar, 0.0)
    return jnp.minimum(jnp.sqrt(jnp.maximum(z, 1e-12)) * model.sigma * v**2,
                       model.vol_cap * model.sigma)


# --- simulations -----------------------------------------------------------------

def simulate():
    sims = {}
    for T in EXPIRIES:
        for b in SKEWS:
            m = base._replace(b=b)
            fn = volscale.bootstrap_fn(curve, m, grid)
            y_d, _ = terminal_yield(m, curve, T, fn, KEY, T + DELTA, N)
            y_t, _ = terminal_yield(m, curve, T, fn, KEY, float(T), N)
            y_q, d_q = terminal_yield(m, curve, T, fn, KEY, "Q", N)
            y_p, d_p = terminal_yield(m, curve, T, fn, KEY, "P", N)
            sims[(T, b)] = {"T+d": (y_d, None), "T": (y_t, None), "Q-nodisc": (y_q, None),
                            "P": (y_p, None), "P-disc": (y_p, d_p)}
            for name, lv in (("D1", lv_variance_linear), ("D2", lv_squared)):
                prev = use_local_vol(lv)
                try:
                    # the bootstrap scaling is the library's: the error is in the dynamics only
                    sims[(T, b)][name] = (terminal_yield(m, curve, T, fn, KEY, T + DELTA, N)[0], None)
                finally:
                    use_local_vol(prev)
        print(f"simulated {T}y", flush=True)
    return sims


# --- hypotheses ------------------------------------------------------------------

def hypotheses():
    H = []
    for dist, strikes in itertools.product(("T+d", "T", "Q-nodisc", "P", "P-disc"), ("curve", "mean")):
        H.append(dict(family="P", dist=dist, fwd="mean", rule="any", strikes=strikes, scale=None))
        for fwd, rule in itertools.product(("curve", "spot"), ("call", "put", "otm", "itm")):
            H.append(dict(family="P", dist=dist, fwd=fwd, rule=rule, strikes=strikes, scale=None))
    for scale, fwd in itertools.product(("N1", "N2"), ("mean", "curve")):
        H.append(dict(family="N", dist="T+d", fwd=fwd, rule="otm", strikes="curve", scale=scale))
    for d in ("D1", "D2"):
        H.append(dict(family="D", dist=d, fwd="mean", rule="any", strikes="mean", scale=None))
    return H


def name(h):
    s = f"{h['family']}: {h['dist']} / inv {h['fwd']} / {h['rule']} / K from {h['strikes']}"
    return s + (f" / {h['scale']}" if h["scale"] else "")


def smile(h, sim, T):
    y, w = sim[h["dist"]]
    w = np.ones_like(y) if w is None else w / w.mean()
    mean = float(np.mean(w * y))
    Fc = float(curve.fwd_yield(T, T + DELTA))
    Finv = {"mean": mean, "curve": Fc, "spot": float(curve.fwd_yield(0.0, DELTA))}[h["fwd"]]
    K = (Fc if h["strikes"] == "curve" else mean) + OFFS
    c = {None: 1.0, "N1": float(curve.discount(T + DELTA) / curve.discount(T)),
         "N2": float(curve.discount(T))}[h["scale"]]
    out = []
    for k in K:
        put = {"call": False, "put": True, "any": False, "otm": k < Finv, "itm": k > Finv}[h["rule"]]
        pay = np.maximum(k - y, 0.0) if put else np.maximum(y - k, 0.0)
        out.append(implied(c * float(np.mean(w * pay)), Finv, k, T, put))
    return BP * np.array(out)


GENUINE_ABSENT_20Y = {0.0: (160, 200), 0.5: (-200, -160), 1.0: (-200, -160, -120)}


def score(sm):
    err = {}
    for T in EXPIRIES:
        e = []
        for b in SKEWS:
            p = np.array(fig4[str(T)][f"b={b:.2f}"], dtype=float)
            ok = ~np.isnan(p) & ~np.isnan(sm[(T, b)])
            e.extend(np.abs(sm[(T, b)] - p)[ok])
        err[T] = float(np.mean(e)) if e else float("nan")
    s0, s1 = sm[(20, 0.0)], sm[(20, 1.0)]
    atm0, atm1 = s0[I[0.0]], s1[I[0.0]]
    far = [s0[I[o]] for o in (160.0, 200.0)]
    sig = {
        "S1": bool(s0[I[120.0]] <= atm0 - 8 and all(np.isnan(v) or v <= atm0 - 20 for v in far)),
        "S2": bool(atm1 - atm0 > 2),
        "S3": bool(s1[I[200.0]] - s1[I[-80.0]] >= 80),
        "S4": bool(all(np.isnan(s1[I[o]]) for o in (-200.0, -160.0, -120.0))),
    }
    miss = sum(int(not np.isnan(sm[(20, b)][I[float(o)]])) for b, os_ in GENUINE_ABSENT_20Y.items() for o in os_)
    shape = all(sig.values())
    verdict = ("reproduces" if shape and all(err[T] <= 3 for T in EXPIRIES)
               else "shape only" if shape else "refuted")
    return dict(err=err, sig=sig, absent_but_model_has=miss, verdict=verdict,
                atm_change_20y=float(atm1 - atm0), slope_20y=float(s1[I[200.0]] - s1[I[-80.0]]))


def batch_se(h, sims, nb=8):
    """MC standard error of the 20y smiles for one hypothesis, from nb batches
    (antithetic pairs kept together)."""
    out = {}
    half = N // 2
    for b in SKEWS:
        reps = []
        for j in range(nb):
            idx = np.r_[j * half // nb:(j + 1) * half // nb]
            idx = np.concatenate([idx, idx + half])
            sub = {k: (v[0][idx], None if v[1] is None else v[1][idx]) for k, v in sims[(20, b)].items()}
            reps.append(smile(h, sub, 20))
        out[b] = (np.nanstd(np.array(reps), axis=0, ddof=1) / np.sqrt(nb)).tolist()
    return out


if __name__ == "__main__":
    sims = simulate()
    H = hypotheses()
    print(f"{len(H)} hypotheses, {N} paths", flush=True)
    rows = []
    for h in H:
        sm = {(T, b): smile(h, sims[(T, b)], T) for T in EXPIRIES for b in SKEWS}
        r = score(sm)
        rows.append(dict(name=name(h), **h, **r,
                         smiles_20y={str(b): sm[(20, b)].tolist() for b in SKEWS}))
    rows.sort(key=lambda r: (r["verdict"] != "reproduces", r["verdict"] != "shape only",
                             -sum(r["sig"].values()), r["err"][20]))
    print("\nverdict      S1 S2 S3 S4  err 5/10/15/20y        ATMchg slope  hypothesis")
    for r in rows:
        s = " ".join("x " if r["sig"][k] else ". " for k in ("S1", "S2", "S3", "S4"))
        e = "/".join(f"{r['err'][T]:4.1f}" for T in EXPIRIES)
        print(f"{r['verdict']:12s} {s} {e:22s} {r['atm_change_20y']:+6.1f} {r['slope_20y']:5.1f}  {r['name']}")
    counts = {v: sum(r["verdict"] == v for r in rows) for v in ("reproduces", "shape only", "refuted")}
    nsig = {k: sum(r["sig"][k] for r in rows) for k in ("S1", "S2", "S3", "S4")}
    print(f"\n{counts}; signature pass counts {nsig}")
    best = rows[0]
    hb = {k: best[k] for k in ("family", "dist", "fwd", "rule", "strikes", "scale")}
    se = batch_se(hb, sims)
    print(f"MC s.e. of the 20y smiles for the top row (bp, max over strikes): "
          + ", ".join(f"b={b}: {np.nanmax(v):.2f}" for b, v in se.items()))
    save("a1_bug_search", dict(n_paths=N, n_hypotheses=len(H), counts=counts, signature_counts=nsig,
                               top_row_se_20y={str(k): v for k, v in se.items()}, rows=rows))

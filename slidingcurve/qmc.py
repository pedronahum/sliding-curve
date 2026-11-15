"""Randomised quasi-Monte Carlo normals: scrambled Sobol points with a
Brownian-bridge construction.

The standard construction for path simulation (Glasserman, 2004, Section
5.5; also the one usually recommended for interest-rate models): the first,
best-distributed Sobol coordinates go to the large-scale shape of each
Brownian path (its end point, then the midpoints), the later coordinates to
the fine detail. The drivers are interleaved, so the rate Brownian motion
and the variance Brownian motion share the leading coordinates.

Each batch is an independently scrambled (Owen) Sobol set, so the batch
means are independent and unbiased: the spread across batches is an honest
error estimate (randomised QMC). The output plugs into `simulate(...,
normals=...)` and `simulate_terminal(..., normals=...)`, in place of the
pseudo-random draws, with the same per-step layout.
"""
import numpy as np
from scipy.stats import norm, qmc


def bridge_plan(n):
    """Construction order of a Brownian bridge on n equal steps: for each new
    point (index k of W at step k, 1..n), its left and right neighbours
    already built, the interpolation weights and the conditional sd (in units
    of sqrt(dt))."""
    plan = [(n, 0, None, 0.0, 1.0, np.sqrt(n))]          # W_n = W_0 + sqrt(n) z
    queue = [(0, n)]
    while queue:
        nxt = []
        for left, right in queue:
            if right - left < 2:
                continue
            mid = (left + right) // 2
            a, b = mid - left, right - mid
            plan.append((mid, left, right, b / (a + b), a / (a + b), np.sqrt(a * b / (a + b))))
            nxt += [(left, mid), (mid, right)]
        queue = nxt
    return plan


def _bridge_increments(z, plan, n):
    """z: [n, P] standard normals in construction order -> [n, P] increments
    of a standard Brownian motion on n unit steps."""
    W = np.zeros((n + 1, z.shape[1]))
    for k, (idx, left, right, wl, wr, sd) in enumerate(plan):
        base = wl * W[left] + (wr * W[right] if right is not None else 0.0)
        W[idx] = base + sd * z[k]
    return np.diff(W, axis=0)


def sobol_normals(n_steps, n_batches, per_batch, n_drivers=2, seed=0, bridge=True):
    """Standard normals of shape [n_steps, n_drivers, n_batches, per_batch]:
    the step increments of n_drivers Brownian motions divided by sqrt(dt).

    per_batch must be a power of two (Sobol balance); each batch is a
    separately scrambled point set. bridge=False assigns coordinates to time
    steps in order (plain Sobol), for comparison."""
    m = int(np.log2(per_batch))
    if 2**m != per_batch:
        raise ValueError("per_batch must be a power of two")
    d = n_steps * n_drivers
    plan = bridge_plan(n_steps)
    out = np.empty((n_steps, n_drivers, n_batches, per_batch))
    rng = np.random.default_rng(seed)
    for k in range(n_batches):
        u = qmc.Sobol(d, scramble=True, seed=rng).random_base2(m)       # [P, d]
        z = norm.ppf(np.clip(u, 1e-16, 1 - 1e-16)).T                   # [d, P]
        for j in range(n_drivers):
            zj = z[j::n_drivers]                                         # interleaved coordinates
            out[:, j, k, :] = _bridge_increments(zj, plan, n_steps) if bridge else zj
    return out

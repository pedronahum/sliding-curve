"""B2: paths split across forced host devices (run in a subprocess, since the
device count is fixed when jax starts)."""
import os
import subprocess
import sys
import textwrap


def test_sharded_paths_keep_antithetic_pairs_and_agree_within_monte_carlo_error():
    code = textwrap.dedent("""
        import numpy as np, jax
        from slidingcurve import Model, markets, pricing as pr, volscale
        assert jax.device_count() == 4
        c = markets.paper_curve("market")
        g = Model(0.03, 5.0, 0.012)
        fn = lambda t: volscale.ybar_zeroth(c, 5.0, t)
        n = 2**14
        x1 = pr.simulate_terminal(g, c, 20.0, fn, jax.random.PRNGKey(0), n, 24, 25.0).x
        pr.set_path_devices(4)
        x4 = pr.simulate_terminal(g, c, 20.0, fn, jax.random.PRNGKey(0), n, 24, 25.0).x
        # Gaussian: x_i + x_{i + n/2} is (up to the 1e-6 noise QE leaves in z at
        # eta = 0) the same number for every antithetic pair; a misaligned
        # pairing would spread it by the size of x itself (~0.05)
        s, s1 = (np.asarray(a[: n // 2] + a[n // 2:]) for a in (x4, x1))
        assert np.ptp(s) < 2 * np.ptp(s1) + 1e-12 and np.ptp(s) < 1e-4
        assert abs(s.mean() - s1.mean()) < 1e-6
        m = Model(0.03, 5.0, 0.012, b=0.4, eta=0.5)
        f2 = volscale.bootstrap_fn(c, m, np.arange(1.0, 21.0))
        off = np.array([-0.02, 0.0, 0.02])
        _, p4, se4 = pr.yield_option_prices(m, c, 20.0, off, f2, jax.random.PRNGKey(1), n)
        pr.set_path_devices(1)
        _, p1, se1 = pr.yield_option_prices(m, c, 20.0, off, f2, jax.random.PRNGKey(1), n)
        assert np.all(np.abs(np.asarray(p4 - p1)) < 4 * np.sqrt(np.asarray(se1)**2 + np.asarray(se4)**2))
        assert not np.allclose(np.asarray(p4), np.asarray(p1))      # different random numbers
        print("ok")
    """)
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]

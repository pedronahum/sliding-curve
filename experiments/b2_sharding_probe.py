"""B2: speed-up from splitting paths across CPU cores (forced host devices).
Run as: XLA_FLAGS=--xla_force_host_platform_device_count=N python b2_sharding_probe.py"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from slidingcurve import Model, markets, volscale, pricing as pr

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

import inspect
NOCHECK = {"check_vma": False} if "check_vma" in inspect.signature(shard_map).parameters else {"check_rep": False}
curve = markets.paper_curve("market")
m = Model(0.03, 5.0, 0.0119, b=0.35)
fn = volscale.bootstrap_fn(curve, m, np.arange(1.0, 21.0))
n = 2**16
devs = jax.devices()
mesh = Mesh(np.array(devs), ("d",))


def one(key):
    state = pr.simulate_terminal(m, curve, 20.0, fn, key, n // len(devs), 24, 25.0)
    return state.x, state.y


@jax.jit
def sharded(key):
    keys = jax.random.split(key, len(devs))
    f = shard_map(lambda k: tuple(a[None] for a in one(k[0])), mesh=mesh, in_specs=P("d"), out_specs=(P("d"), P("d")), **NOCHECK)
    return f(keys)


@jax.jit
def single(key):
    return pr.simulate_terminal(m, curve, 20.0, fn, key, n, 24, 25.0)


for name, f in (("single device", single), (f"{len(devs)} devices", sharded)):
    jax.block_until_ready(f(jax.random.PRNGKey(0)))
    t0 = time.perf_counter()
    for i in range(3):
        jax.block_until_ready(f(jax.random.PRNGKey(i)))
    print(f"{name:12s}: {(time.perf_counter() - t0) / 3:.3f}s per simulation of {n} paths x 480 steps")

"""Type aliases shared across the package.

Not part of the public API; import the names from here rather than repeating
the unions, so that the signatures stay readable and consistent:

    Array       a materialised jax array (what the package returns)
    ArrayLike   anything `jnp.asarray` accepts (what the package takes); this
                is jax's own alias, so it excludes Python sequences
    FloatSeq    a nested sequence of floats, for the public entry points that
                also accept plain Python lists
    ZeroYieldFn the scalar function a `Curve` is built from

The package is float64 throughout (see `__init__`), so no dtype parameter is
exposed and none is annotated here.
"""
from collections.abc import Callable, Sequence
from typing import Any, TypeAlias

import jax
import jax.typing

Array: TypeAlias = jax.Array
ArrayLike: TypeAlias = jax.typing.ArrayLike
FloatSeq: TypeAlias = Sequence[float] | Sequence[Sequence[float]]
ZeroYieldFn: TypeAlias = Callable[[Any], Any]
# The volatility scaling Ybar_1(t) the simulators take (see volscale.bootstrap_fn).
YbarFn: TypeAlias = Callable[[ArrayLike], Array]

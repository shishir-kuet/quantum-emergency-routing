"""Reproducible random-number streams.

Experiments must be repeatable, but different components (instance sampling,
simulated annealing, QAOA parameter initialisation) should not share a single
mutable generator -- otherwise adding a call in one place silently changes the
numbers everywhere else.

:func:`make_rng` solves this by deriving an independent, deterministic stream
from the master seed plus a set of string tags.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

__all__ = ["make_rng", "derive_seed"]

_MASK_64 = (1 << 64) - 1


def derive_seed(seed: int, *tags: str) -> int:
    """Deterministically mix *seed* with *tags* into a fresh 64-bit seed.

    The mixing uses BLAKE2b rather than :func:`hash`, because Python's string
    hashing is randomised per process and would break reproducibility.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(seed).to_bytes(8, "little", signed=False))
    for tag in tags:
        digest.update(b"\x00")
        digest.update(str(tag).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") & _MASK_64


def make_rng(seed: int, *tags: str) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` for the given seed and tags.

    Examples
    --------
    >>> a = make_rng(42, "instance")
    >>> b = make_rng(42, "instance")
    >>> bool((a.random(5) == b.random(5)).all())
    True
    >>> c = make_rng(42, "annealing")
    >>> bool((make_rng(42, "instance").random(5) == c.random(5)).all())
    False
    """
    return np.random.default_rng(derive_seed(seed, *tags))


def seed_sequence(seed: int, count: int, *tags: str) -> Iterable[int]:
    """Yield *count* independent integer seeds derived from *seed* and *tags*."""
    for index in range(count):
        yield derive_seed(seed, *tags, f"#{index}")

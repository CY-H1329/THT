"""Seeded RNG with named substreams.

A single master seed produces deterministic sub-streams for independent
subsystems (layout, names, images, enemies, ...) so that adding a stochastic
draw in one subsystem does not perturb another.
"""

import hashlib
import random


def _derive(seed: int, label: str) -> int:
    """Derive a 64-bit substream seed from (master seed, label)."""
    h = hashlib.blake2b(f"{seed}:{label}".encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


class EnvRNG:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    # --- proxy methods --------------------------------------------------
    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)

    def uniform(self, a: float, b: float) -> float:
        return self._rng.uniform(a, b)

    def choice(self, seq):
        return self._rng.choice(seq)

    def choices(self, population, weights=None, k: int = 1):
        """Weighted sampling with replacement. Mirrors random.choices."""
        return self._rng.choices(population, weights=weights, k=k)

    def sample(self, population, k: int):
        return self._rng.sample(population, k)

    def shuffle(self, x):
        self._rng.shuffle(x)

    def random(self) -> float:
        return self._rng.random()

    # --- substream factory ---------------------------------------------
    def substream(self, label: str) -> "EnvRNG":
        return EnvRNG(_derive(self.seed, label))

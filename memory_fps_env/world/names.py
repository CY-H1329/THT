"""Room-name sources: public pool, held-out pool, procedural generator."""

from pathlib import Path
from typing import List
from memory_fps_env.rng import EnvRNG

_PUBLIC_PATH = Path(__file__).resolve().parent.parent / "assets" / "names_public.txt"

# Word lists for procedural name generation.
_ADJECTIVES = [
    "Crimson", "Azure", "Ivory", "Onyx", "Verdant", "Amber", "Slate",
    "Russet", "Cobalt", "Pearl", "Saffron", "Ebon", "Mauve", "Garnet",
    "Indigo", "Topaz", "Coral", "Jade", "Vermilion", "Opal",
    "Velvet", "Ashen", "Glacial", "Sable", "Citrine", "Umber",
]
_NOUNS = [
    "Hall", "Atrium", "Wing", "Vault", "Court", "Gallery", "Chamber",
    "Foyer", "Annex", "Vestibule", "Loft", "Cellar", "Parlor",
    "Solarium", "Pavilion", "Library", "Refectory", "Conservatory",
    "Salon", "Belfry",
]
_CONSONANTS = list("bcdfghjklmnprstvwz")
_VOWELS = list("aeiou")


class NameSource:
    def sample(self, rng: EnvRNG, k: int) -> List[str]:
        raise NotImplementedError


class PublicNameSource(NameSource):
    def __init__(self, path: Path | None = None):
        path = path or _PUBLIC_PATH
        with open(path, "r", encoding="utf-8") as f:
            self.pool = [line.strip() for line in f if line.strip()]

    def sample(self, rng: EnvRNG, k: int) -> List[str]:
        if k > len(self.pool):
            raise ValueError(f"Requested {k} names, pool size {len(self.pool)}")
        return rng.sample(self.pool, k)


class HeldoutNameSource(NameSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def sample(self, rng: EnvRNG, k: int) -> List[str]:
        with open(self.path, "r", encoding="utf-8") as f:
            pool = [line.strip() for line in f if line.strip()]
        if k > len(pool):
            raise ValueError(f"Requested {k} names, held-out pool size {len(pool)}")
        return rng.sample(pool, k)


class ProceduralNameSource(NameSource):
    def __init__(self, mode: str = "adj_noun"):
        if mode not in {"adj_noun", "nonce"}:
            raise ValueError(f"Unknown procedural mode: {mode}")
        self.mode = mode

    def sample(self, rng: EnvRNG, k: int) -> List[str]:
        if self.mode == "adj_noun":
            return self._sample_adj_noun(rng, k)
        return self._sample_nonce(rng, k)

    def _sample_adj_noun(self, rng: EnvRNG, k: int) -> List[str]:
        seen, out = set(), []
        while len(out) < k:
            name = f"{rng.choice(_ADJECTIVES)} {rng.choice(_NOUNS)}"
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def _sample_nonce(self, rng: EnvRNG, k: int) -> List[str]:
        seen, out = set(), []
        while len(out) < k:
            syllables = rng.randint(2, 3)
            stem = "".join(
                rng.choice(_CONSONANTS) + rng.choice(_VOWELS)
                for _ in range(syllables)
            )
            name = stem.capitalize() + " " + rng.choice(_NOUNS)
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

"""Which law a reference chain points to.

The right context of a chain (up to a sentence end or the next chain) is
cut into prefixes of 1..max_tokens words. Every prefix and every alias
from law_aliases.json is lemmatized and embedded as TF-IDF over character
3-5-grams; the prefix-alias pair with the highest cosine similarity gives
the law if it is above `threshold`. Character n-grams tolerate inflection
left by the lemmatizer, typos, "КоАП" without "РФ", "Закона N 27-ФЗ" and
old titles of a law; a date or number between the chain and the name is
just a longer prefix.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from law_links.chains import Chain
from law_links.lemmas import Lemmatizer, tokenize
from law_links.normalize import normalize

# The right context stops at a sentence end, a semicolon or the next chain.
_STOP_RE = re.compile(
    r"(?<=[а-яa-z0-9)\"])\.\s+(?=[А-ЯA-Z])|;"
    r"|(?<![0-9a-zа-я])(?:ст|стать[а-я]*|п|пп|пункт[а-я]*|ч|част[а-я]*)\.?\s*\d",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LawMention:
    law_id: int
    start: int
    end: int
    candidates: Tuple[int, ...] = ()


class LawResolver:
    def __init__(
        self,
        aliases: Dict[str, List[str]],
        threshold: float = 0.85,
        ngram_range: Tuple[int, int] = (3, 5),
        max_tokens: int = 20,
        max_chars: int = 250,
        lemmatizer: Optional[Lemmatizer] = None,
    ) -> None:
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.lemmatizer = lemmatizer or Lemmatizer()
        self.alias_law: List[int] = []
        self.alias_keys: List[str] = []
        for law_id, names in aliases.items():
            for name in names:
                self.alias_keys.append(self.lemmatizer.text(normalize(name)))
                self.alias_law.append(int(law_id))
        self.alias_law_arr = np.array(self.alias_law)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
        self.alias_matrix = self.vectorizer.fit_transform(self.alias_keys)
        self._alias_t = self.alias_matrix.T.tocsr()  # transposed once, not per call
        exact: Dict[str, set] = {}
        for key, law in zip(self.alias_keys, self.alias_law):
            exact.setdefault(key, set()).add(law)
        self._exact = {key: tuple(sorted(laws)) for key, laws in exact.items()}
        self._best: Dict[str, Tuple[float, Tuple[int, ...]]] = {}

    @classmethod
    def from_json(cls, path: Path, **kw) -> "LawResolver":
        return cls(json.loads(Path(path).read_text("utf-8")), **kw)

    def similarity(self, texts: List[str]) -> np.ndarray:
        """Cosine similarity [text x alias] of lemma strings."""
        return (self.vectorizer.transform(texts) @ self._alias_t).toarray()

    def best_aliases(self, keys: List[str]) -> List[Tuple[float, Tuple[int, ...]]]:
        """Top similarity and the laws reaching it, per lemma string; cached,
        since law names repeat from text to text."""
        todo = [k for k in dict.fromkeys(keys) if k not in self._best]
        if todo:
            sims = self.similarity(todo)
            if len(self._best) > 200_000:
                self._best.clear()
            for key, row in zip(todo, sims):
                top = float(row.max())
                laws = tuple(sorted(set(self.alias_law_arr[np.flatnonzero(row >= top - 1e-9)].tolist())))
                self._best[key] = (top, laws)
        return [self._best[k] for k in keys]

    def prefixes(self, text: str, chain: Chain) -> List[Tuple[int, int, str]]:
        """Prefixes of the chain's right context: (start, end, lemma string)."""
        window = text[chain.end : chain.end + self.max_chars]
        stop = _STOP_RE.search(window)
        if stop:
            window = window[: stop.start()]
        tokens = tokenize(window)[: self.max_tokens]
        lemmas = [self.lemmatizer.key(t) for t in tokens]
        return [
            (chain.end + tokens[0].start, chain.end + tokens[k - 1].end, " ".join(lemmas[:k]))
            for k in range(1, len(tokens) + 1)
        ]

    def resolve(self, text: str, chain: Chain) -> Optional[LawMention]:
        cands = self.prefixes(text, chain)
        if not cands:
            return None
        # A prefix equal to an alias has similarity 1.0, the maximum: the
        # shortest such prefix is what the full search would pick.
        for start, end, key in cands:
            if key in self._exact:
                laws = self._exact[key]
                return LawMention(laws[0], start, end, laws)
        scored = self.best_aliases([c[2] for c in cands])
        best_c = max(range(len(cands)), key=lambda i: (scored[i][0], -i))
        top, laws = scored[best_c]
        if top < self.threshold:
            return None
        start, end, _ = cands[best_c]
        return LawMention(laws[0], start, end, laws)

    def lookup(self, name: str) -> List[int]:
        """Laws whose alias has exactly the same lemmas as `name`."""
        key = self.lemmatizer.text(normalize(name))
        return sorted({law for law, k in zip(self.alias_law, self.alias_keys) if k == key})

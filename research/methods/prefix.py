"""Linking by similarity of the chain's right context to law aliases.

For every chain the right context (up to a sentence end or the next chain)
is cut into prefixes of 1..max_tokens tokens; every prefix is compared with
every alias, the best pair above `threshold` gives the law. A PrefixLinker
replaces both the mention finder and the nearest-mention rule; subclasses
differ only in the similarity function.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from law_links.aliases import LawMention, Lemmatizer, tokenize
from law_links.grammar import Chain
from law_links.normalize import normalize

# Right context stops at a sentence end or at the next chain marker.
_STOP_RE = re.compile(
    r"(?<=[а-яa-z0-9)\"])\.\s+(?=[А-ЯA-Z])|;|(?<![0-9a-zа-я])(?:ст|стать[а-я]*|п|пп|пункт[а-я]*|ч|част[а-я]*)\.?\s*\d",
    re.IGNORECASE,
)


class LemmaText:
    """Lower-cased lemmas joined by spaces, short upper abbreviations kept."""

    def __init__(self, lemmatizer: Optional[Lemmatizer] = None) -> None:
        self.lemmatizer = lemmatizer or Lemmatizer()

    def __call__(self, text: str) -> str:
        return " ".join(self.lemmatizer.key(t) for t in tokenize(text))


class PrefixLinker:
    """Scores prefixes of the chain's right context against all aliases.
    Subclasses implement `similarity(prefixes) -> matrix [prefix x alias]`."""

    def __init__(
        self,
        aliases_path: Path,
        threshold: float,
        max_tokens: int = 20,
        max_chars: int = 250,
        lemmatize: bool = True,
    ) -> None:
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.lemma = LemmaText() if lemmatize else (lambda s: " ".join(t.text.lower() for t in tokenize(s)))
        raw = json.loads(Path(aliases_path).read_text("utf-8"))
        self.alias_law: List[int] = []
        self.alias_docs: List[str] = []
        for law_id, names in raw.items():
            for name in names:
                self.alias_docs.append(self.lemma(normalize(name)))
                self.alias_law.append(int(law_id))
        self.alias_law_arr = np.array(self.alias_law)

    def similarity(self, prefixes: List[str]) -> np.ndarray:
        raise NotImplementedError

    def candidates(self, text: str, chain: Chain) -> List[Tuple[int, int, str]]:
        """Prefixes of the right context: (start, end, lemma string)."""
        window = text[chain.end : chain.end + self.max_chars]
        stop = _STOP_RE.search(window)
        if stop:
            window = window[: stop.start()]
        tokens = tokenize(window)[: self.max_tokens]
        out = []
        for k in range(1, len(tokens) + 1):
            span = window[: tokens[k - 1].end]
            out.append((chain.end + tokens[0].start, chain.end + tokens[k - 1].end, self.lemma(span)))
        return out

    def link(self, text: str, chain: Chain, mentions: Sequence[LawMention], starts) -> Optional[LawMention]:
        cands = self.candidates(text, chain)
        if not cands:
            return None
        sims = self.similarity([c[2] for c in cands])
        best_c, best_a = np.unravel_index(np.argmax(sims), sims.shape)
        top = sims[best_c, best_a]
        if top < self.threshold:
            return None
        laws = sorted(set(self.alias_law_arr[np.flatnonzero(sims[best_c] >= top - 1e-9)].tolist()))
        start, end, _ = cands[best_c]
        return LawMention(laws[0], start, end, tuple(laws))

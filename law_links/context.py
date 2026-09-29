"""Steps that use the text around a reference.

AnaphoraResolver: "ст. 19.5 названного Кодекса", "статьей 20 настоящего
Федерального закона" name no law; such a chain takes the closest law of
the same kind (code or other act) linked earlier in the text.

ContextNB: a law number shared by several acts ("ФЗ №63-ФЗ" is both 246 and
396) is resolved by multinomial naive Bayes. Every candidate law is a class
whose word distribution is estimated from its aliases (add-k smoothing),
the evidence is the words around the reference, the prior is uniform.
Words are cut to 5-letter stems: a title and the text around a reference
often share only a root ("адвокатом" / "адвокатской деятельности").
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from law_links.lemmas import Lemmatizer, tokenize
from law_links.normalize import normalize
from law_links.resolver import LawMention

# A law word needs a pointing word before it ("настоящего Федерального
# закона"); a bare "Кодекса" is accepted only when no name follows it.
_ANAPHORA_RE = re.compile(
    r"^[\s,]*(?:(?:названн|указанн|настоящ|эт|данн|того\s+же|вышеуказанн|вышеназванн)[а-я]*\s+"
    r"(?P<kind>кодекс|федеральн[а-я]*\s+закон|закон)[а-я]*"
    r"|(?P<bare>кодекс)[а-я]*(?=\s*[,.;)]|\s+[а-я]))",
    re.IGNORECASE,
)
_STOPWORDS = {"о", "об", "в", "и", "на", "с", "по", "для", "от", "росси", "федер", "рф", "закон", "фз"}


class AnaphoraResolver:
    def __init__(self, aliases: Dict[str, List[str]]) -> None:
        self.codes = {int(k) for k, names in aliases.items() if any("кодекс" in n.lower() for n in names)}

    def resolve(self, text: str, chain_end: int, previous: Sequence[Optional[LawMention]]) -> Optional[LawMention]:
        match = _ANAPHORA_RE.match(text[chain_end : chain_end + 60])
        if not match:
            return None
        want_code = (match.group("kind") or match.group("bare")).lower().startswith("кодекс")
        for mention in reversed(previous):
            if mention is not None and (mention.law_id in self.codes) == want_code:
                return LawMention(mention.law_id, chain_end + match.start(), chain_end + match.end(), (mention.law_id,))
        return None


class ContextNB:
    def __init__(
        self,
        aliases: Dict[str, List[str]],
        window_chars: int = 320,
        k: float = 0.1,
        stem: int = 5,
        lemmatizer: Optional[Lemmatizer] = None,
    ) -> None:
        self.window_chars = window_chars
        self.k = k
        self.stem = stem
        self.lemmatizer = lemmatizer or Lemmatizer()
        self.counts: Dict[int, Counter] = {}
        vocab = set()
        for law_id, names in aliases.items():
            counts = Counter()
            for name in names:
                counts.update(self.stems(name))
            self.counts[int(law_id)] = counts
            vocab.update(counts)
        self.vocab_size = len(vocab) + 1

    def stems(self, text: str) -> List[str]:
        stems = (self.lemmatizer.key(t)[: self.stem] for t in tokenize(normalize(text)))
        return [s for s in stems if s not in _STOPWORDS]

    def choose(self, text: str, start: int, end: int, candidates: Sequence[int]) -> int:
        """The candidate with the highest likelihood of the context words.
        Words unknown to every candidate carry no evidence and are skipped."""
        words = self.stems(text[max(0, start - self.window_chars) : end + self.window_chars])
        words = [w for w in words if any(w in self.counts[c] for c in candidates)]
        best, best_score = candidates[0], -math.inf
        for law in candidates:
            counts = self.counts[law]
            denom = sum(counts.values()) + self.k * self.vocab_size
            score = sum(math.log((counts[w] + self.k) / denom) for w in words)
            if score > best_score:
                best, best_score = law, score
        return best

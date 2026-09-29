"""Context steps applied after linking.

AnaphoraResolver: "ст. 19.5 названного Кодекса", "статьей 20 настоящего
Федерального закона" name no law; such a chain takes the closest law of
the same kind (code or other act) linked earlier in the text.

ContextNB: a law number shared by several acts ("ФЗ №63-ФЗ" is both 246
and 396) is resolved by multinomial naive Bayes: each candidate law is a
class whose word distribution comes from the lemmas of its aliases, the
evidence is the lemmas around the reference, the prior is uniform.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from law_links.aliases import LawMention, Lemmatizer, tokenize
from law_links.normalize import normalize

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
    def __init__(self, aliases_path: Path) -> None:
        raw: Dict[str, List[str]] = json.loads(Path(aliases_path).read_text("utf-8"))
        self.codes = {int(k) for k, names in raw.items() if any("кодекс" in n.lower() for n in names)}

    def resolve(self, text: str, chain_end: int, previous: Sequence[Optional[LawMention]]) -> Optional[LawMention]:
        match = _ANAPHORA_RE.match(text[chain_end : chain_end + 60])
        if not match:
            return None
        kind = match.group("kind") or match.group("bare")
        want_code = kind.lower().startswith("кодекс")
        for mention in reversed(previous):
            if mention is not None and (mention.law_id in self.codes) == want_code:
                return LawMention(mention.law_id, chain_end + match.start(), chain_end + match.end(), (mention.law_id,))
        return None


class ContextNB:
    def __init__(self, aliases_path: Path, window: int = 40, k: float = 0.1, stem: int = 5) -> None:
        self.window = window
        self.stem = stem
        self.k = k
        self.lemmatizer = Lemmatizer()
        raw: Dict[str, List[str]] = json.loads(Path(aliases_path).read_text("utf-8"))
        self.counts: Dict[int, Counter] = {}
        vocab = set()
        for law_id, names in raw.items():
            c = Counter()
            for name in names:
                c.update(w for w in self.lemmas(name) if w not in _STOPWORDS)
            self.counts[int(law_id)] = c
            vocab.update(c)
        self.v = len(vocab) + 1

    def lemmas(self, text: str) -> List[str]:
        """Crude stems (first `stem` letters of the lemma): the words of a
        title and of the text around a reference often share only a root
        ("адвокатом" / "адвокатской деятельности")."""
        return [self.lemmatizer.key(t)[: self.stem] for t in tokenize(normalize(text))]

    def choose(self, text: str, start: int, end: int, candidates: Sequence[int]) -> int:
        context = text[max(0, start - self.window * 8) : end + self.window * 8]
        words = [w for w in self.lemmas(context) if w not in _STOPWORDS]
        best, best_score = candidates[0], -math.inf
        for law in candidates:
            c = self.counts[law]
            total = sum(c.values())
            score = sum(math.log((c[w] + self.k) / (total + self.k * self.v)) for w in words if w in c)
            # words absent from every candidate carry no evidence; unseen-in-this-class words are penalized
            score += sum(
                math.log(self.k / (total + self.k * self.v))
                for w in words
                if w not in c and any(w in self.counts[o] for o in candidates)
            )
            if score > best_score:
                best, best_score = law, score
        return best

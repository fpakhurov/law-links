"""Which law a reference chain points to.

The right context of a chain (up to a sentence end or the next chain) is
cut into prefixes of 1..max_tokens words. Every prefix and every alias
from law_aliases.json is lemmatized and embedded as TF-IDF over character
3-5-grams; the prefix-alias pair with the highest cosine similarity gives
the law if it is above `threshold`. Character n-grams tolerate inflection
left by the lemmatizer, typos, "КоАП" without "РФ", "Закона N 27-ФЗ" and
old titles of a law; a date or number between the chain and the name is
just a longer prefix. A word pymorphy does not know that is one edit away
from a form of an alias word is corrected first ("Трудовго" -> "Трудового").
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from law_links.chains import Chain
from law_links.lemmas import Lemmatizer, Token, edits1, morph, tokenize
from law_links.normalize import normalize

# The right context stops at a sentence end, a semicolon or the next chain.
_STOP_RE = re.compile(
    r"(?<=[а-яa-z0-9)\"])\.\s+(?=[А-ЯA-Z])|;"
    r"|(?<![0-9a-zа-я])(?:ст|стать[а-я]*|п|пп|пункт[а-я]*|ч|част[а-я]*)\.?\s*\d",
    re.IGNORECASE,
)
_SENTENCE_END_RE = re.compile(r"(?<=[а-яa-z0-9)\"])\.\s+(?=[А-ЯA-Z])|;")
# A law name before a chain must have a document word in it.
_DOC_WORD_RE = re.compile(
    r"(?i:кодекс|закон|конституц|указ|постановлени|регламент|\bфз\b)|\b(?:[А-Я]{1,3}К\s+РФ|КоАП)\b"
)
# The name of some document right after a chain.
_DOC_AFTER_RE = re.compile(
    r"[\s,]*(?:федеральн|закон|кодекс|конституц|постановлени|указ|приказ|распоряжени|положени|правил"
    r"|регламент|инструкц|устав|договор|соглашени|конвенц|приложени|решени)",
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
        self._forms = set()
        for name in {t.text.lower() for names in aliases.values() for n in names for t in tokenize(normalize(n))}:
            if name.isalpha() and len(name) >= 4:
                self._forms.update(f.word for f in morph().parse(name)[0].lexeme)
        self._fixed: Dict[str, str] = {}

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
        tokens = [Token(self.spell(t.text), t.start, t.end) for t in tokenize(window)[: self.max_tokens]]
        lemmas = [self.lemmatizer.key(t) for t in tokens]
        return [
            (chain.end + tokens[0].start, chain.end + tokens[k - 1].end, " ".join(lemmas[:k]))
            for k in range(1, len(tokens) + 1)
        ]

    def spell(self, word: str) -> str:
        """The alias word form one edit away from an unknown word, or the word."""
        if len(word) < 5 or not word.isalpha():
            return word
        if word not in self._fixed:
            low = word.lower()
            fixed = word
            if low not in self._forms and not morph().word_is_known(low):
                cands = sorted(edits1(low) & self._forms)
                if cands:
                    fixed = cands[0].upper() if word.isupper() else cands[0].capitalize() if word[0].isupper() else cands[0]
            if len(self._fixed) > 200_000:
                self._fixed.clear()
            self._fixed[word] = fixed
        return self._fixed[word]

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

    def resolve_left(self, text: str, chain: Chain, start: int) -> Optional[LawMention]:
        """A law named before a chain in the same sentence ("Согласно
        Уголовному кодексу РФ ... статьёй 105"). Every word span of
        text[start:chain.start] after the last sentence end that has a
        document word ("кодекс", "закон", "УК РФ") is matched to the aliases;
        the best span wins, the one closest to the chain on a tie. Short
        alias titles ("О связи") are too common as plain words to be taken
        without one. A chain followed by a document name unknown to the
        aliases ("статьи 25 Федерального закона ...") is left alone."""
        if _DOC_AFTER_RE.match(text, chain.end):
            return None
        start = max(start, chain.start - self.max_chars)
        window = text[start : chain.start]
        for stop in _SENTENCE_END_RE.finditer(window):
            start, window = start + stop.end(), window[stop.end() :]
        tokens = [Token(self.spell(t.text), t.start, t.end) for t in tokenize(window)][-self.max_tokens :]
        lemmas = [self.lemmatizer.key(t) for t in tokens]
        spans = [
            (start + tokens[i].start, start + tokens[j - 1].end, " ".join(lemmas[i:j]))
            for i in range(len(tokens))
            for j in range(i + 1, len(tokens) + 1)
            if _DOC_WORD_RE.search(window, tokens[i].start, tokens[j - 1].end)
        ]
        if not spans:
            return None
        scored = self.best_aliases([s[2] for s in spans])
        best = max(range(len(spans)), key=lambda i: (scored[i][0], spans[i][1], -spans[i][0]))
        top, laws = scored[best]
        if top < self.threshold:
            return None
        return LawMention(laws[0], spans[best][0], spans[best][1], laws)

    def lookup(self, name: str) -> List[int]:
        """Laws whose alias has exactly the same lemmas as `name`."""
        key = self.lemmatizer.text(normalize(name))
        return sorted({law for law, k in zip(self.alias_law, self.alias_keys) if k == key})

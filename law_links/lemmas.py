"""Word tokens and lemmas for matching law names.

Words are lemmatized with pymorphy3 (first parse), short upper-case
abbreviations ("НК", "РФ") are kept as they are, Latin look-alike letters
in mixed-script words are folded to Cyrillic ("CК" -> "СК").
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

import pymorphy3

TOKEN_RE = re.compile(r"[0-9a-zа-я]+(?:[-.][0-9a-zа-я]+)*", re.IGNORECASE)
SHORT_ABBR_MAX_LEN = 5
_LATIN_TO_CYR = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")
_HAS_CYR = re.compile(r"[а-яА-Я]")
_HAS_LAT = re.compile(r"[a-zA-Z]")


def fold_homoglyphs(word: str) -> str:
    """Replace Latin look-alike letters in mixed-script words ("CК" -> "СК")."""
    if _HAS_CYR.search(word) and _HAS_LAT.search(word):
        return word.translate(_LATIN_TO_CYR)
    return word


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def is_short_abbr(self) -> bool:
        return self.text.isupper() and len(self.text) <= SHORT_ABBR_MAX_LEN


def tokenize(text: str) -> List[Token]:
    return [Token(fold_homoglyphs(m.group()), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


class Lemmatizer:
    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        self._cached = lru_cache(maxsize=200_000)(self._lemma)

    def _lemma(self, word: str) -> str:
        lower = word.lower()
        if not lower.isalpha():
            return lower
        return self._morph.parse(lower)[0].normal_form

    def key(self, token: Token) -> str:
        if token.is_short_abbr:
            return token.text.lower()
        return self._cached(token.text)

    def text(self, text: str) -> str:
        """Lemmas of all words joined by spaces."""
        return " ".join(self.key(t) for t in tokenize(text))

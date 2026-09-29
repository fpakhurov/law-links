"""Law name index: finds mentions of laws in text regardless of declension.

Aliases from law_aliases.json are tokenized and lemmatized with pymorphy3,
then stored in a trie keyed by lemmas. Text tokens are matched against the
trie using all their candidate lemmas, longest match wins.

Build once at startup, then `find_mentions` is read-only and thread-safe.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymorphy3

from law_links.normalize import normalize

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[0-9a-zа-я]+(?:[-.][0-9a-zа-я]+)*", re.IGNORECASE)
# Characters allowed between two tokens of one law name. A period is not
# allowed, so a match never crosses a sentence boundary.
GAP_RE = re.compile(r'^[\s"№(),\-]*$')
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


@dataclass(frozen=True)
class LawMention:
    law_id: int
    start: int
    end: int
    candidates: Tuple[int, ...] = ()


@dataclass
class _Terminal:
    law_id: int
    upper_mask: Tuple[int, ...]
    alias: str


@dataclass
class _Node:
    children: Dict[str, "_Node"] = field(default_factory=dict)
    terminals: List[_Terminal] = field(default_factory=list)


def tokenize(text: str) -> List[Token]:
    return [
        Token(fold_homoglyphs(m.group()), m.start(), m.end())
        for m in TOKEN_RE.finditer(text)
    ]


class Lemmatizer:
    def __init__(self) -> None:
        self._morph = pymorphy3.MorphAnalyzer()
        self._cache_candidates = lru_cache(maxsize=200_000)(self._candidates)

    def _candidates(self, word: str) -> Tuple[str, ...]:
        lower = word.lower()
        if not lower.isalpha():
            return (lower,)
        seen: List[str] = []
        for parse in self._morph.parse(lower):
            if parse.normal_form not in seen:
                seen.append(parse.normal_form)
        if lower not in seen:
            seen.append(lower)
        return tuple(seen)

    def candidates(self, token: Token) -> Tuple[str, ...]:
        if token.is_short_abbr:
            return (token.text.lower(),)
        return self._cache_candidates(token.text)

    def is_regular_word(self, word: str) -> bool:
        """True for dictionary words that are not abbreviations ("как")."""
        lower = word.lower()
        if not self._morph.word_is_known(lower):
            return False
        return "Abbr" not in self._morph.parse(lower)[0].tag

    def key(self, token: Token) -> str:
        return self.candidates(token)[0]


class AliasIndex:
    def __init__(self, lemmatizer: Optional[Lemmatizer] = None) -> None:
        self._lemmatizer = lemmatizer or Lemmatizer()
        self._root = _Node()
        self.n_aliases = 0
        self.n_conflicts = 0

    @classmethod
    def from_json(cls, path: Path) -> "AliasIndex":
        with open(path, "r", encoding="utf-8") as file:
            raw: Dict[str, List[str]] = json.load(file)
        index = cls()
        for law_id, aliases in raw.items():
            for alias in aliases:
                index.add(int(law_id), alias)
        index.n_conflicts = index._count_conflicts(index._root)
        logger.info(
            "Alias index built: laws=%d aliases=%d ambiguous_keys=%d",
            len(raw),
            index.n_aliases,
            index.n_conflicts,
        )
        return index

    def add(self, law_id: int, alias: str) -> None:
        tokens = tokenize(normalize(alias))
        if not tokens:
            return
        node = self._root
        for token in tokens:
            node = node.children.setdefault(self._lemmatizer.key(token), _Node())
        mask = tuple(i for i, t in enumerate(tokens) if t.is_short_abbr)
        node.terminals.append(_Terminal(law_id, mask, alias))
        self.n_aliases += 1

    def lookup(self, alias: str) -> List[int]:
        """Exact lookup of a law name, used by tests and the audit script."""
        text = normalize(alias)
        tokens = tokenize(text)
        mention = self._match_at(text, tokens, 0)
        if mention is None or mention[0] != len(tokens):
            return []
        return list(mention[1])

    def find_mentions(self, text: str) -> List[LawMention]:
        tokens = tokenize(text)
        mentions: List[LawMention] = []
        i = 0
        while i < len(tokens):
            match = self._match_at(text, tokens, i)
            if match is None:
                i += 1
                continue
            end, law_ids = match
            mentions.append(
                LawMention(
                    law_id=min(law_ids),
                    start=tokens[i].start,
                    end=tokens[end - 1].end,
                    candidates=tuple(sorted(law_ids)),
                )
            )
            i = end
        return mentions

    def _match_at(
        self, text: str, tokens: List[Token], start: int
    ) -> Optional[Tuple[int, Tuple[int, ...]]]:
        """Return (end_token_index, law_ids) of the longest match at `start`."""
        frontier = [self._root]
        best: Optional[Tuple[int, Tuple[int, ...]]] = None
        j = start
        while frontier and j < len(tokens):
            if j > start:
                if not GAP_RE.match(text[tokens[j - 1].end : tokens[j].start]):
                    break
            next_frontier: List[_Node] = []
            for node in frontier:
                for cand in self._lemmatizer.candidates(tokens[j]):
                    child = node.children.get(cand)
                    if child is not None and child not in next_frontier:
                        next_frontier.append(child)
            frontier = next_frontier
            j += 1
            law_ids = set()
            for node in frontier:
                for term in node.terminals:
                    if all(self._abbr_ok(tokens[start + k]) for k in term.upper_mask):
                        law_ids.add(term.law_id)
            if law_ids:
                best = (j, tuple(sorted(law_ids)))
        return best

    def _abbr_ok(self, token: Token) -> bool:
        """Short upper-case aliases ("НК", "БК") match lower case text only
        when the word is not a regular dictionary word."""
        return token.text.isupper() or not self._lemmatizer.is_regular_word(token.text)

    def _count_conflicts(self, node: _Node) -> int:
        own = 1 if len({t.law_id for t in node.terminals}) > 1 else 0
        return own + sum(self._count_conflicts(c) for c in node.children.values())

    def iter_terminals(self):
        stack: List[Tuple[_Node, Tuple[str, ...]]] = [(self._root, ())]
        while stack:
            node, path = stack.pop()
            if node.terminals:
                yield path, node.terminals
            for key, child in node.children.items():
                stack.append((child, path + (key,)))

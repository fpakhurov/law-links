"""Composable extractor for experiments.

An extractor is split into three replaceable parts:
    ChainFinder:   text -> reference chains ("пп. 1 п. 2 ст. 3")
    MentionFinder: text -> law mentions with law_id
    Linker:        chain + mentions -> the mention the chain refers to
Every part gets the normalized text, offsets are shared.
"""

import bisect
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from law_links.aliases import AliasIndex, LawMention, tokenize
from law_links.extractor import _FILLER_RE, expand
from law_links.grammar import Chain, find_chains
from law_links.normalize import normalize
from law_links.schemas import LawLink


class ChainFinder(Protocol):
    def find(self, text: str) -> List[Chain]: ...


class MentionFinder(Protocol):
    def find(self, text: str) -> List[LawMention]: ...


class Linker(Protocol):
    def link(
        self, text: str, chain: Chain, mentions: Sequence[LawMention], starts: Sequence[int]
    ) -> Optional[LawMention]: ...


# Between chains of one enumeration: "ч. 6 ст. 15, ст.ст. 64 и 73 УК РФ".
_ENUM_GAP_RE = re.compile(r"^[\s,;]*(?:(?:и|а также|также|так и|так и по|и по)[\s,]*)?$", re.IGNORECASE)


class Pipeline:
    """Optional steps after linking:
    anaphora   "ст. 19.5 названного Кодекса" takes the last code linked before
    propagate  a chain without a law takes the law of the next chain when
               only commas or "и" stand between them
    context_nb a law number shared by several acts is resolved by context
    """

    def __init__(
        self,
        chains: ChainFinder,
        mentions: MentionFinder,
        linker: Linker,
        propagate: bool = False,
        anaphora=None,
        context_nb=None,
    ) -> None:
        self.chains = chains
        self.mentions = mentions
        self.linker = linker
        self.propagate = propagate
        self.anaphora = anaphora
        self.context_nb = context_nb

    def extract(self, text: str) -> List[LawLink]:
        norm = normalize(text)
        mentions = sorted(self.mentions.find(norm), key=lambda m: m.start)
        starts = [m.start for m in mentions]
        chains = self.chains.find(norm)
        laws = [self.linker.link(norm, chain, mentions, starts) for chain in chains]
        if self.context_nb is not None:
            for i, m in enumerate(laws):
                if m is not None and len(m.candidates) > 1:
                    law = self.context_nb.choose(norm, chains[i].start, m.end, m.candidates)
                    laws[i] = LawMention(law, m.start, m.end, m.candidates)
        if self.anaphora is not None:
            for i, chain in enumerate(chains):
                if laws[i] is None:
                    laws[i] = self.anaphora.resolve(norm, chain.end, laws[:i])
        if self.propagate:
            for i in range(len(chains) - 2, -1, -1):
                if laws[i] is None and laws[i + 1] is not None:
                    if _ENUM_GAP_RE.match(norm[chains[i].end : chains[i + 1].start]):
                        laws[i] = laws[i + 1]
        links: List[LawLink] = []
        for chain, mention in zip(chains, laws):
            if mention is not None:
                links.extend(expand(chain, mention.law_id))
        return links


# Rule-based parts (the current service).


class RegexChains:
    def find(self, text: str) -> List[Chain]:
        return find_chains(text)


class TrieMentions:
    """Lemma trie over law aliases (law_links.aliases.AliasIndex)."""

    def __init__(self, index: AliasIndex) -> None:
        self.index = index

    def find(self, text: str) -> List[LawMention]:
        return self.index.find_mentions(text)


class NearestRightLinker:
    """First mention to the right; only filler words may stand between."""

    def link(self, text, chain, mentions, starts):
        i = bisect.bisect_left(starts, chain.end)
        if i == len(mentions):
            return None
        mention = mentions[i]
        if _FILLER_RE.match(text[chain.end : mention.start]):
            return mention
        return None


# Naive baseline parts.


class ExactMentions:
    """Exact case-insensitive match of alias token sequences, no morphology.
    Finds only aliases written in the dictionary form ("ГК РФ", "Налоговый
    кодекс РФ"), not inflected ones ("Налогового кодекса")."""

    def __init__(self, aliases: Dict[str, List[str]]) -> None:
        self.table: Dict[Tuple[str, ...], List[int]] = {}
        for law_id, names in aliases.items():
            for name in names:
                key = tuple(t.text.lower() for t in tokenize(normalize(name)))
                if key:
                    self.table.setdefault(key, []).append(int(law_id))
        self.max_len = max(len(k) for k in self.table)

    @classmethod
    def from_json(cls, path: Path) -> "ExactMentions":
        return cls(json.loads(Path(path).read_text("utf-8")))

    def find(self, text: str) -> List[LawMention]:
        tokens = tokenize(text)
        words = [t.text.lower() for t in tokens]
        result: List[LawMention] = []
        i = 0
        while i < len(tokens):
            for n in range(min(self.max_len, len(tokens) - i), 0, -1):
                ids = self.table.get(tuple(words[i : i + n]))
                if ids:
                    result.append(
                        LawMention(min(ids), tokens[i].start, tokens[i + n - 1].end, tuple(sorted(set(ids))))
                    )
                    i += n
                    break
            else:
                i += 1
        return result

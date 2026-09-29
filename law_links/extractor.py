"""Link extractors.

RuleBasedExtractor pairs every reference chain with the nearest law mention
to its right, allowing a few filler words between them
("ст. 16 Закона "О защите прав потребителей"").
"""

import bisect
import itertools
import re
from dataclasses import dataclass
from typing import List, Optional, Protocol

from law_links.aliases import AliasIndex, LawMention
from law_links.grammar import Chain, find_chains
from law_links.normalize import normalize
from law_links.schemas import LawLink

MAX_FILLER_WORDS = 3
# Words allowed between a chain and a law name that the alias may not cover:
# "ст. 16 Закона "О ..."", "ст. 12 Федерального закона "О ..."".
_FILLER_WORD = (
    r"(?:закон[а-я]*|федеральн[а-я]*|фз|кодекс[а-я]*|указ[а-я]*|"
    r"постановлени[а-я]*|приказ[а-я]*|рф|росси[а-я]*|российск[а-я]*|"
    r"федераци[а-я]*|президент[а-я]*|правительств[а-я]*)"
)
_FILLER_RE = re.compile(
    rf'^[\s",]*(?:{_FILLER_WORD}[\s",]*){{0,{MAX_FILLER_WORDS}}}$', re.IGNORECASE
)


class LinkExtractor(Protocol):
    def extract(self, text: str) -> List[LawLink]: ...


@dataclass(frozen=True)
class DetectedLink:
    link: LawLink
    start: int
    end: int
    law_candidates: tuple


class RuleBasedExtractor:
    def __init__(self, index: AliasIndex) -> None:
        self.index = index

    def extract(self, text: str) -> List[LawLink]:
        return [d.link for d in self.extract_detailed(text)]

    def extract_detailed(self, text: str) -> List[DetectedLink]:
        norm = normalize(text)
        mentions = self.index.find_mentions(norm)
        starts = [m.start for m in mentions]
        result: List[DetectedLink] = []
        for chain in find_chains(norm):
            mention = self._law_for_chain(norm, chain, mentions, starts)
            if mention is None:
                continue
            for link in expand(chain, mention.law_id):
                result.append(
                    DetectedLink(link, chain.start, mention.end, mention.candidates)
                )
        return result

    @staticmethod
    def _law_for_chain(
        text: str, chain: Chain, mentions: List[LawMention], starts: List[int]
    ) -> Optional[LawMention]:
        i = bisect.bisect_left(starts, chain.end)
        if i == len(mentions):
            return None
        mention = mentions[i]
        if _FILLER_RE.match(text[chain.end : mention.start]):
            return mention
        return None


def expand(chain: Chain, law_id: int) -> List[LawLink]:
    return [
        LawLink(law_id=law_id, article=a, point_article=p, subpoint_article=s)
        for a, p, s in itertools.product(
            chain.articles or [None],
            chain.points or [None],
            chain.subpoints or [None],
        )
    ]

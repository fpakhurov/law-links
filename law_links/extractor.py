"""Extraction of legal references.

Steps:
1. normalize the text (length preserving, offsets stay valid);
2. find reference chains with the HMM tagger (law_links.chains);
3. resolve the law of every chain by TF-IDF similarity of its right
   context to law aliases (law_links.resolver);
4. a law number shared by several acts is chosen by naive Bayes over the
   context; a chain followed by "названного Кодекса" takes the last code
   (law_links.context);
5. in an enumeration ("ч. 6 ст. 15, ст.ст. 64 и 73 УК РФ") a chain without
   a law takes the law of the next chain;
6. lists of values are expanded into separate links.
"""

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

from law_links.chains import Chain, ChainTagger
from law_links.context import AnaphoraResolver, ContextNB
from law_links.lemmas import Lemmatizer
from law_links.normalize import normalize
from law_links.resolver import LawMention, LawResolver
from law_links.schemas import LawLink

# Between chains of one enumeration: ", ", " и ", ", так и по ".
_ENUM_GAP_RE = re.compile(r"^[\s,;]*(?:(?:и|а также|также|так и|так и по|и по)[\s,]*)?$", re.IGNORECASE)


class LinkExtractor(Protocol):
    def extract(self, text: str) -> List[LawLink]: ...


@dataclass(frozen=True)
class DetectedLink:
    link: LawLink
    start: int
    end: int
    law_candidates: Tuple[int, ...]


class Extractor:
    def __init__(
        self,
        chains: ChainTagger,
        resolver: LawResolver,
        anaphora: Optional[AnaphoraResolver] = None,
        context_nb: Optional[ContextNB] = None,
    ) -> None:
        self.chains = chains
        self.resolver = resolver
        self.anaphora = anaphora
        self.context_nb = context_nb

    @classmethod
    def from_files(cls, aliases_path: Path, chain_model_path: Path) -> "Extractor":
        aliases = json.loads(Path(aliases_path).read_text("utf-8"))
        lemmatizer = Lemmatizer()
        return cls(
            ChainTagger.from_file(chain_model_path),
            LawResolver(aliases, lemmatizer=lemmatizer),
            AnaphoraResolver(aliases),
            ContextNB(aliases, lemmatizer=lemmatizer),
        )

    def extract(self, text: str) -> List[LawLink]:
        return [d.link for d in self.extract_detailed(text)]

    def extract_detailed(self, text: str) -> List[DetectedLink]:
        norm = normalize(text)
        chains = self.chains.find(norm)
        laws: List[Optional[LawMention]] = [self.resolver.resolve(norm, c) for c in chains]
        for i, chain in enumerate(chains):
            mention = laws[i]
            if mention is not None and len(mention.candidates) > 1 and self.context_nb is not None:
                law = self.context_nb.choose(norm, chain.start, mention.end, mention.candidates)
                laws[i] = LawMention(law, mention.start, mention.end, mention.candidates)
            elif mention is None and self.anaphora is not None:
                laws[i] = self.anaphora.resolve(norm, chain.end, laws[:i])
        for i in range(len(chains) - 2, -1, -1):
            if laws[i] is None and laws[i + 1] is not None:
                if _ENUM_GAP_RE.match(norm[chains[i].end : chains[i + 1].start]):
                    laws[i] = laws[i + 1]
        result: List[DetectedLink] = []
        for chain, mention in zip(chains, laws):
            if mention is None:
                continue
            for link in expand(chain, mention.law_id):
                result.append(DetectedLink(link, chain.start, mention.end, mention.candidates))
        return result


def expand(chain: Chain, law_id: int) -> List[LawLink]:
    return [
        LawLink(law_id=law_id, article=a, point_article=p, subpoint_article=s)
        for a, p, s in itertools.product(
            chain.articles or [None],
            chain.points or [None],
            chain.subpoints or [None],
        )
    ]

"""Law resolution by fuzzy string matching (lecture 03 practice, fuzzywuzzy).

rapidfuzz implements the same scorers as fuzzywuzzy in C++. Scores are
scaled to 0..1 to share thresholds with the other prefix linkers.
"""

from pathlib import Path
from typing import List

import numpy as np
from rapidfuzz import fuzz, process

from research.methods.prefix import PrefixLinker

SCORERS = {
    "ratio": fuzz.ratio,
    "token_sort": fuzz.token_sort_ratio,
    "token_set": fuzz.token_set_ratio,
    "wratio": fuzz.WRatio,
}


class FuzzyLinker(PrefixLinker):
    def __init__(self, aliases_path: Path, threshold: float = 0.85, scorer: str = "ratio", **kw) -> None:
        super().__init__(aliases_path, threshold, **kw)
        self.scorer = SCORERS[scorer]

    def similarity(self, prefixes: List[str]) -> np.ndarray:
        return process.cdist(prefixes, self.alias_docs, scorer=self.scorer, workers=-1) / 100.0

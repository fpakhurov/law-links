"""Law resolution by TF-IDF vectors and cosine similarity (lecture 03).

Aliases and context prefixes are lemmatized and embedded as TF-IDF over
character n-grams (optionally plus words); similarity is cosine.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from research.methods.prefix import PrefixLinker


class TfidfLinker(PrefixLinker):
    def __init__(
        self,
        aliases_path: Path,
        threshold: float = 0.7,
        ngram_range: Tuple[int, int] = (2, 4),
        word_weight: float = 0.0,
        **kw,
    ) -> None:
        super().__init__(aliases_path, threshold, **kw)
        self.word_weight = word_weight
        docs = self.alias_docs
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, sublinear_tf=True)
        char_m = self.char_vec.fit_transform(docs)
        if word_weight > 0:
            self.word_vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)
            word_m = self.word_vec.fit_transform(docs) * word_weight
            self.alias_m = l2_normalize(hstack([char_m, word_m]).tocsr())
        else:
            self.word_vec = None
            self.alias_m = char_m

    def _embed(self, texts: List[str]):
        char_m = self.char_vec.transform(texts)
        if self.word_vec is None:
            return char_m
        word_m = self.word_vec.transform(texts) * self.word_weight
        return l2_normalize(hstack([char_m, word_m]).tocsr())

    def similarity(self, prefixes: List[str]) -> np.ndarray:
        return (self._embed(prefixes) @ self.alias_m.T).toarray()

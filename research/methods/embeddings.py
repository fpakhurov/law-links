"""Law resolution with dense word vectors (lecture 03: word2vec).

Skip-gram with negative sampling is trained on lemmatized court decisions
(gensim). A prefix or alias is the average of its lemma vectors; the
similarity is cosine. Optionally mixed with the character TF-IDF
similarity: sim = (1 - mix) * tfidf + mix * w2v.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
from gensim.models import Word2Vec

from law_links.aliases import Lemmatizer, tokenize
from law_links.normalize import normalize
from research.corpus import DATA_DIR, training_texts
from research.methods.prefix import PrefixLinker

LEMMA_CACHE = DATA_DIR / "corpus_lemmas.txt"
MODEL_DIR = DATA_DIR / "models"


def lemmatized_corpus(lemmatizer: Lemmatizer) -> List[List[str]]:
    """One lemma list per line of the training corpus, cached on disk."""
    if not LEMMA_CACHE.exists():
        with open(LEMMA_CACHE, "w", encoding="utf-8") as out:
            for doc in training_texts():
                for line in map(normalize, doc.split("\n")):
                    words = [lemmatizer.key(t) for t in tokenize(line)]
                    if len(words) > 1:
                        out.write(" ".join(words) + "\n")
    return [line.split() for line in LEMMA_CACHE.read_text("utf-8").splitlines()]


def train_word2vec(dim: int, window: int, epochs: int, seed: int = 13) -> Word2Vec:
    path = MODEL_DIR / f"w2v_d{dim}_w{window}_e{epochs}.model"
    if path.exists():
        return Word2Vec.load(str(path))
    sentences = lemmatized_corpus(Lemmatizer())
    model = Word2Vec(
        sentences, vector_size=dim, window=window, min_count=2, sg=1, negative=10,
        epochs=epochs, workers=1, seed=seed,
    )
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return model


class Word2VecLinker(PrefixLinker):
    def __init__(
        self,
        aliases_path: Path,
        threshold: float = 0.8,
        dim: int = 100,
        window: int = 5,
        epochs: int = 10,
        mix: float = 1.0,
        **kw,
    ) -> None:
        super().__init__(aliases_path, threshold, **kw)
        self.mix = mix
        self.wv = train_word2vec(dim, window, epochs).wv
        self.alias_m = self._embed(self.alias_docs)
        if mix < 1.0:
            from research.methods.tfidf import TfidfLinker

            self.tfidf = TfidfLinker(aliases_path, threshold=0.0, **kw)

    def _embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.wv.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            vecs = [self.wv[w] for w in text.split() if w in self.wv]
            if vecs:
                v = np.mean(vecs, axis=0)
                out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out

    def similarity(self, prefixes: List[str]) -> np.ndarray:
        sims = self._embed(prefixes) @ self.alias_m.T
        if self.mix < 1.0:
            sims = self.mix * sims + (1 - self.mix) * self.tfidf.similarity(prefixes)
        return sims

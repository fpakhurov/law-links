"""Law resolution with n-gram language models (lecture 02).

Each law gets a word bigram model over the lemmas of its aliases, a
background bigram model is trained on court decisions. A context prefix
w_1..w_k is scored for every law by the log-likelihood ratio

    LLR(law) = sum_i log P_law(w_i | w_{i-1}) - log P_bg(w_i | w_{i-1})

Both models use linear interpolation of bigram, unigram and a uniform
floor, bigram and unigram counts use add-k smoothing. With a uniform prior
over laws argmax LLR is the Bayes decision; the LLR value itself tells
whether the prefix looks like a law name at all (threshold).

The background model stands for "not a law name", so law mentions found
by the alias trie are cut out of the corpus before training it.

Prefix scores are accumulated token by token, so a chain costs
O(tokens x laws).
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from law_links.aliases import AliasIndex, LawMention, Lemmatizer, tokenize
from law_links.grammar import Chain
from law_links.normalize import normalize
from research.corpus import training_texts
from research.methods.prefix import _STOP_RE

BOS = "<s>"


class BigramLM:
    """Interpolated bigram model: l2 * P_bi + l1 * P_uni + l0 / V."""

    def __init__(self, lambdas=(0.6, 0.35, 0.05), k: float = 0.01) -> None:
        self.l2, self.l1, self.l0 = lambdas
        self.k = k
        self.uni: Counter = Counter()
        self.bi: Dict[str, Counter] = defaultdict(Counter)
        self.total = 0
        self.vocab_size = 1

    def fit(self, sentences: Iterable[List[str]], vocab_size: Optional[int] = None) -> "BigramLM":
        for words in sentences:
            prev = BOS
            for w in words:
                self.uni[w] += 1
                self.bi[prev][w] += 1
                prev = w
            self.total += len(words)
        self.vocab_size = vocab_size or len(self.uni) + 1
        return self

    def logprob(self, prev: str, w: str) -> float:
        v = self.vocab_size
        p_uni = (self.uni[w] + self.k) / (self.total + self.k * v)
        row = self.bi.get(prev)
        p_bi = (row[w] + self.k) / (sum(row.values()) + self.k * v) if row else p_uni
        return math.log(self.l2 * p_bi + self.l1 * p_uni + self.l0 / v)


class LawLMs:
    """Per-law bigram models stored as arrays over laws for fast scoring."""

    def __init__(self, law_sentences: Dict[int, List[List[str]]], lambdas, k: float) -> None:
        self.laws = sorted(law_sentences)
        self.l2, self.l1, self.l0 = lambdas
        self.k = k
        vocab = {w for sents in law_sentences.values() for s in sents for w in s}
        self.v = len(vocab) + 1
        n = len(self.laws)
        self.uni: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n))
        self.bi: Dict[Tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(n))
        self.prev_total: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n))
        self.total = np.zeros(n)
        for j, law in enumerate(self.laws):
            for words in law_sentences[law]:
                prev = BOS
                for w in words:
                    self.uni[w][j] += 1
                    self.bi[(prev, w)][j] += 1
                    self.prev_total[prev][j] += 1
                    prev = w
                self.total[j] += len(words)
        self._zeros = np.zeros(n)

    def logprob(self, prev: str, w: str) -> np.ndarray:
        k, v = self.k, self.v
        p_uni = (self.uni.get(w, self._zeros) + k) / (self.total + k * v)
        pt = self.prev_total.get(prev)
        if pt is None:
            p_bi = p_uni
        else:
            p_bi = (self.bi.get((prev, w), self._zeros) + k) / (pt + k * v)
        return np.log(self.l2 * p_bi + self.l1 * p_uni + self.l0 / v)


class NgramLinker:
    def __init__(
        self,
        aliases_path: Path,
        threshold: float = 5.0,
        lambdas=(0.6, 0.35, 0.05),
        k: float = 0.01,
        bg_docs: int = 400,
        max_tokens: int = 20,
        max_chars: int = 250,
    ) -> None:
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.max_chars = max_chars
        self.lemmatizer = Lemmatizer()
        raw = json.loads(Path(aliases_path).read_text("utf-8"))
        law_sents = {
            int(law_id): [self.lemmas(normalize(a)) for a in names] for law_id, names in raw.items()
        }
        self.law_lm = LawLMs(law_sents, lambdas, k)
        self.laws = np.array(self.law_lm.laws)
        index = AliasIndex.from_json(aliases_path)
        bg_sents: List[List[str]] = []
        for doc in training_texts(bg_docs):
            for line in map(normalize, doc.split("\n")):
                pos = 0
                for mention in index.find_mentions(line) + [None]:
                    end = mention.start if mention else len(line)
                    words = self.lemmas(line[pos:end])
                    if words:
                        bg_sents.append(words)
                    pos = mention.end if mention else pos
        self.bg = BigramLM(lambdas, k).fit(bg_sents)

    def lemmas(self, text: str) -> List[str]:
        return [self.lemmatizer.key(t) for t in tokenize(text)]

    def link(self, text: str, chain: Chain, mentions: Sequence[LawMention], starts) -> Optional[LawMention]:
        window = text[chain.end : chain.end + self.max_chars]
        stop = _STOP_RE.search(window)
        if stop:
            window = window[: stop.start()]
        tokens = tokenize(window)[: self.max_tokens]
        if not tokens:
            return None
        score = np.zeros(len(self.laws))
        prev = BOS
        best: Tuple[float, int, int] = (-math.inf, -1, -1)
        for i, token in enumerate(tokens):
            w = self.lemmatizer.key(token)
            score = score + self.law_lm.logprob(prev, w) - self.bg.logprob(prev, w)
            j = int(np.argmax(score))
            if score[j] > best[0]:
                best = (float(score[j]), j, i)
            prev = w
        value, j, i = best
        if value < self.threshold:
            return None
        return LawMention(
            int(self.laws[j]), chain.end + tokens[0].start, chain.end + tokens[i].end, (int(self.laws[j]),)
        )

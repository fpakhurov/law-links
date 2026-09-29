"""Chain tagging with a hidden Markov model (lectures 02: Markov assumption,
MLE with add-k smoothing, interpolation) and with a linear-chain CRF.

HMM: P(tags, tokens) = prod P(t_i | t_{i-1}[, t_{i-2}]) * P(shape(w_i) | t_i).
Emissions are over token shapes, so unseen numbers and words are covered:
numbers, dotted numbers, ranges, single letters, capitalized and ordinary
words become classes, reference markers and function words stay as is.
Decoding is Viterbi in log space; the second order model interpolates
trigram, bigram and unigram tag probabilities.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from research.methods.tagging import TAGS

_KEEP_RE = re.compile(r"^(?:ст|стать[а-я]*|пп?|пункт[а-я]*|подп|подпункт[а-я]*|ч|част[а-я]*|пунт[а-я]*)$")
_FUNCTION = {"и", "в", "как", "так", "а", "также", "или", "по", "с", "со", "к", "на", "от", "№", "n"}
_ORD_RE = re.compile(r"^(перв|втор|трет|четверт|пят|шест|седьм|восьм|девят|десят)[а-я]+$")


def shape(token: str) -> str:
    low = token.lower()
    if _KEEP_RE.match(low) or low in _FUNCTION:
        return low
    if _ORD_RE.match(low):
        return "<ORD>"
    if re.fullmatch(r"\d{1,4}", token):
        return "<INT>"
    if re.fullmatch(r"\d{5,}", token):
        return "<LONGINT>"
    if re.fullmatch(r"\d+(?:\.\d+)+", token):
        return "<DEC>"
    if re.fullmatch(r"\d+(?:\.\d+)*\s*-\s*\d+(?:\.\d+)*", token):
        return "<RANGE>"
    if len(token) == 1 and token.isalpha():
        return "<LET>"
    if token.isalpha():
        if token.isupper() and len(token) <= 5:
            return "<ABBR>"
        return "<CAP>" if token[0].isupper() else "<W>"
    return token


class HMMTagger:
    def __init__(self, order: int = 1, k: float = 0.01, lambdas=(0.6, 0.3, 0.1)) -> None:
        self.order = order
        self.k = k
        self.lambdas = lambdas
        self.tags = TAGS
        self.n = len(TAGS)

    def fit(self, data: Sequence[Tuple[List[str], List[str]]]) -> "HMMTagger":
        idx = {t: i for i, t in enumerate(self.tags)}
        n = self.n
        start = np.zeros(n)  # index n stands for the sentence start
        uni = np.zeros(n)
        bi = np.zeros((n + 1, n))
        tri = np.zeros((n + 1, n + 1, n))
        emit: Dict[int, Counter] = defaultdict(Counter)
        vocab = set()
        for tokens, tags in data:
            prev2, prev1 = n, n
            for tok, tag in zip(tokens, tags):
                j = idx[tag]
                s = shape(tok)
                vocab.add(s)
                emit[j][s] += 1
                uni[j] += 1
                bi[prev1, j] += 1
                tri[prev2, prev1, j] += 1
                prev2, prev1 = prev1, j
        self.vocab_size = len(vocab) + 1
        k = self.k
        self.log_uni = np.log((uni + k) / (uni.sum() + k * n))
        self.p_uni = np.exp(self.log_uni)
        self.p_bi = (bi + k) / (bi.sum(axis=1, keepdims=True) + k * n)
        self.p_tri = (tri + k) / (tri.sum(axis=2, keepdims=True) + k * n)
        self.emit_totals = np.array([sum(emit[j].values()) for j in range(n)], dtype=float)
        self.emit = emit
        return self

    def _log_emit(self, s: str) -> np.ndarray:
        counts = np.array([self.emit[j][s] for j in range(self.n)], dtype=float)
        return np.log((counts + self.k) / (self.emit_totals + self.k * self.vocab_size))

    def predict(self, tokens: List[str]) -> List[str]:
        if self.order == 1:
            return self._viterbi1(tokens)
        return self._viterbi2(tokens)

    def _viterbi1(self, tokens: List[str]) -> List[str]:
        n = self.n
        log_bi = np.log(self.p_bi)
        score = log_bi[n] + self._log_emit(shape(tokens[0]))
        back = []
        for tok in tokens[1:]:
            cand = score[:, None] + log_bi[:n]
            back.append(cand.argmax(axis=0))
            score = cand.max(axis=0) + self._log_emit(shape(tok))
        best = [int(score.argmax())]
        for ptr in reversed(back):
            best.append(int(ptr[best[-1]]))
        return [self.tags[i] for i in reversed(best)]

    def _trans2(self) -> np.ndarray:
        """log P(t | t2, t1) interpolated, shape (n+1, n+1, n)."""
        l3, l2, l1 = self.lambdas
        p = l3 * self.p_tri + l2 * self.p_bi[None, :, :] + l1 * self.p_uni[None, None, :]
        return np.log(p)

    def _viterbi2(self, tokens: List[str]) -> List[str]:
        n = self.n
        trans = self._trans2()
        # state = (t_{i-1}, t_i); index n means sentence start
        score = np.full((n + 1, n), -math.inf)
        score[n] = trans[n, n] + self._log_emit(shape(tokens[0]))
        backs = []
        for tok in tokens[1:]:
            e = self._log_emit(shape(tok))
            # new[t1, t] = max_{t2} score[t2, t1] + trans[t2, t1, t]
            cand = score[:, :, None] + trans[:, :n, :]  # (t2, t1, t)
            back = cand.argmax(axis=0)  # (t1, t)
            new = np.full((n + 1, n), -math.inf)
            new[:n] = cand.max(axis=0) + e[None, :]
            score = new
            backs.append(back)
        t1, t = np.unravel_index(int(np.argmax(score)), score.shape)
        path = [int(t), int(t1)]
        for back in reversed(backs[1:]):
            t2 = int(back[path[-1], path[-2]])
            path.append(t2)
        path = list(reversed(path))
        if len(tokens) == 1:
            path = path[-1:]
        return [self.tags[i] for i in path[-len(tokens):]]


class CRFTagger:
    """Linear-chain CRF (sklearn-crfsuite) with window features."""

    def __init__(self, c1: float = 0.05, c2: float = 0.05, max_iterations: int = 150) -> None:
        import sklearn_crfsuite

        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=True
        )

    @staticmethod
    def features(tokens: List[str]) -> List[Dict[str, object]]:
        shapes = [shape(t) for t in tokens]
        feats = []
        for i, tok in enumerate(tokens):
            f: Dict[str, object] = {
                "bias": 1.0,
                "shape": shapes[i],
                "low": tok.lower() if len(tok) < 15 else "<long>",
                "suf3": tok.lower()[-3:],
                "upper": tok[:1].isupper(),
            }
            for d in (-3, -2, -1, 1, 2, 3):
                j = i + d
                f[f"shape{d}"] = shapes[j] if 0 <= j < len(tokens) else "<pad>"
            for d in (-1, 1):
                j = i + d
                f[f"low{d}"] = tokens[j].lower() if 0 <= j < len(tokens) and len(tokens[j]) < 15 else "<pad>"
            feats.append(f)
        return feats

    def fit(self, data: Sequence[Tuple[List[str], List[str]]]) -> "CRFTagger":
        self.model.fit([self.features(t) for t, _ in data], [tags for _, tags in data])
        return self

    def predict(self, tokens: List[str]) -> List[str]:
        return list(self.model.predict_single(self.features(tokens)))

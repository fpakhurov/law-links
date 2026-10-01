"""Reference chains ("пп. 1 п. 2 ст. 3") found by a hidden Markov model.

Tokens are tagged with roles:
    O           outside a chain
    MS VS       subpoint marker / value      ("пп.", "подпункт" / "1", "б")
    MP VP       point or part marker / value ("п.", "ч.", "пункта" / "2", "первой")
    MD VD       dropped level: a part above a point, a paragraph
    MA VA       article marker / value       ("ст.", "статьи" / "3", "19.5")
    X           filler inside a chain        (".", ",", "и", quotes, "в")
by a second order HMM: P(t_i | t_{i-2}, t_{i-1}) interpolates trigram,
bigram and unigram tag estimates, emissions P(shape(w_i) | t_i) are over
token shapes (numbers, dotted numbers, ranges, letters, markers), both with
add-k smoothing; decoding is Viterbi in log space. Tag sequences are then
grouped into Chain objects.

The model is trained offline on synthetic chains inserted into real
sentences (research/train_chains.py) and stored in data/chain_hmm.json.
"""

import json
import math
import re
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from law_links.lemmas import edits1, morph
from law_links.normalize import normalize

TAGS = ["O", "MS", "VS", "MP", "VP", "MD", "VD", "MA", "VA", "X"]
TOKEN_RE = re.compile(r"\d+(?:\.\d+)*(?:\s*-\s*\d+(?:\.\d+)*)?|[А-Яа-яЁёA-Za-z]+|\S")

_KEEP_RE = re.compile(
    r"^(?:ст|стать[а-я]*|пп?|пункт[а-я]*|подп|подпункт[а-я]*|ч|част[а-я]*|пунт[а-я]*|абз|абзац[а-я]*)$"
)
_FUNCTION = {"и", "в", "как", "так", "а", "также", "или", "по", "с", "со", "к", "на", "от", "№", "n"}
_ORD_RE = re.compile(r"^(перв|втор|трет|четверт|пят|шест|седьм|восьм|девят|десят)[а-я]+$")
_ORDINALS = {
    "перв": "1", "втор": "2", "трет": "3", "четверт": "4", "пят": "5",
    "шест": "6", "седьм": "7", "восьм": "8", "девят": "9", "десят": "10",
}


@dataclass(frozen=True)
class Chain:
    start: int
    end: int
    articles: List[str]
    points: List[str]
    subpoints: List[str]


@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int


# A marker glued to the previous word: "спп. 1", "дляст. 105", "Согласностатье 16".
# An abbreviation needs a dot and a number after it, a full word a number.
_GLUED_RE = re.compile(
    r"([а-я]+?)(ст|ч|пп|п|стать[яиеюй][а-я]{0,2}|част[ьияею][а-я]{0,2}|(?:под)?пункт[а-я]{0,3})",
    re.IGNORECASE,
)
_GLUE_HEADS = {"с", "со", "в", "во", "к", "ко", "по", "на", "из", "от", "и"}
_AFTER_ABBR_RE = re.compile(r"\.\s*\d")
_AFTER_WORD_RE = re.compile(r"\s*\d")


def _unglue(word: str, text: str, end: int) -> Optional[int]:
    """Length of the head word if `word` is a word with a marker glued to it."""
    if len(word) < 2 or not word.isalpha() or _KEEP_RE.match(word.lower()):
        return None
    m = _GLUED_RE.fullmatch(word)
    if not m or (len(m.group(1)) < 3 and m.group(1).lower() not in _GLUE_HEADS):
        return None
    after = _AFTER_ABBR_RE if len(m.group(2)) <= 2 else _AFTER_WORD_RE
    return len(m.group(1)) if after.match(text, end) else None


def tokenize(text: str) -> List[Tok]:
    tokens = []
    for m in TOKEN_RE.finditer(text):
        word, start, end = m.group(), m.start(), m.end()
        head = _unglue(word, text, end)
        if head:
            tokens += [Tok(word[:head], start, start + head), Tok(word[head:], start + head, end)]
        else:
            tokens.append(Tok(word, start, end))
    return tokens


_MARKER_FORMS = [
    stem + ending
    for stems, endings in (
        (["стать"], ["я", "и", "е", "ю", "ей", "ям", "ями", "ях"]),
        (["стат"], ["ей"]),
        (["част"], ["ь", "и", "ью", "ей", "ям", "ями", "ях"]),
        (["пункт", "подпункт"], ["", "а", "у", "ом", "е", "ы", "ов", "ам", "ами", "ах"]),
        (["абзац"], ["", "а", "у", "ем", "е", "ы", "ев", "ам", "ами", "ах"]),
    )
    for stem in stems
    for ending in endings
]
_TYPO_TO_MARKER: Dict[str, str] = {}


def is_marker(token: str) -> bool:
    """A level marker word, abbreviation or a marker with a typo."""
    low = token.lower()
    return bool(_KEEP_RE.match(low)) or low in _MARKER_FORMS or _marker_typo(low) is not None


def _marker_typo(low: str) -> Optional[str]:
    """The marker form one edit away from an unknown word: "сттаьи" -> "статьи"."""
    if not _TYPO_TO_MARKER:
        for form in reversed(_MARKER_FORMS):
            for typo in edits1(form):
                _TYPO_TO_MARKER[typo] = form
    form = _TYPO_TO_MARKER.get(low)
    if form is None or len(low) < 4 or morph().word_is_known(low):
        return None
    return form


@lru_cache(maxsize=200_000)
def shape(token: str) -> str:
    """Emission class of a token: markers and function words stay as is,
    a marker with a typo is the marker."""
    low = token.lower()
    if _KEEP_RE.match(low) or low in _FUNCTION:
        return low
    if low == "статей":  # plural genitive, an ordinary word in the synthetic data
        return "статьях"
    if low.isalpha() and 4 <= len(low) <= 11:
        form = _marker_typo(low)
        if form is not None:
            return shape(form)
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


def value_text(token: str) -> str:
    """Normalized value: no spaces in ranges, ordinal words to digits."""
    lower = token.lower()
    for stem, digit in _ORDINALS.items():
        if lower.startswith(stem) and len(lower) > len(stem):
            return digit
    return re.sub(r"\s+", "", lower)


def decode(tokens: Sequence[Tok], tags: Sequence[str]) -> List[Chain]:
    """Group tagged tokens into chains. A chain ends at an O token or when a
    new marker follows an article value and starts another chain: another
    article marker ("ст. 15 ст. 64"), or a lower level followed by its own
    article ("ч. 6 ст. 15 ч. 2 ст. 64"). A lower level after the article with
    no article after it belongs to that article ("Статья 20.4 ч. 1 КоАП").
    A chain without an article whose value is an ordinal word is a part of
    a code ("части второй Кодекса"), not a reference, and is dropped.
    Values listed after one marker share the level of the first of them: in
    "ст. 486, 487., 516" the dot after 487 does not make 516 a subpoint. The
    marker itself does not set the level: "п." is a point or a subpoint.
    A chain without a marker word (numbers in a table, "т. 10, л.д. 5") is
    dropped."""
    tags = list(tags)
    level = None
    for i, tag in enumerate(tags):
        if tag == "O" or tag.startswith("M"):
            level = None
        elif tag.startswith("V"):
            if level is None:
                level = tag[1]
            else:
                tags[i] = "V" + level
    chains: List[Chain] = []
    cur: Optional[dict] = None
    # article_ahead[i]: an article marker follows position i before the chain
    # ends. A comma or semicolon followed by a new marker ends the look
    # ("ст. 12.8 ч. 1, ст. 12.26"); inside a list of values it does not
    # ("пунктами «а», «б» части 2 статьи 105").
    next_marker = [False] * (len(tags) + 1)
    for i in range(len(tags) - 1, -1, -1):
        next_marker[i] = tags[i].startswith("M") or (tags[i] == "X" and next_marker[i + 1])
    article_ahead = [False] * (len(tags) + 1)
    for i in range(len(tags) - 1, -1, -1):
        separator = tags[i] == "O" or (tags[i] == "X" and tokens[i].text in ",;" and next_marker[i + 1])
        article_ahead[i] = not separator and (tags[i] == "MA" or article_ahead[i + 1])

    def close():
        nonlocal cur
        if cur and cur["VD"] and not cur["VS"]:
            # a part is dropped only above a point and a subpoint: "пунктом 2
            # части 1 статьи 5" is point 2 of part 1, "частью второй статьи 96" part 2
            cur["VS"], cur["VP"] = cur["VP"], cur["VD"]
        if cur and cur["marker"] and (cur["VA"] or not cur["ordinal"]) and (cur["VA"] or cur["VP"] or cur["VS"]):
            chains.append(Chain(cur["start"], cur["end"], cur["VA"], cur["VP"], cur["VS"]))
        cur = None

    for i, (tok, tag) in enumerate(zip(tokens, tags)):
        if tag == "O":
            close()
            continue
        if tag.startswith("M") and cur is not None and cur["VA"]:
            if tag == "MA" or article_ahead[i]:
                close()
        if cur is None:
            if tag == "X":
                continue
            # "статей 228, 229" is tagged O V V: the marker may precede the chain
            before = i > 0 and is_marker(tokens[i - 1].text)
            cur = {"start": tok.start, "end": tok.end, "VA": [], "VP": [], "VS": [], "VD": [], "ordinal": False,
                   "marker": before, "part": False}
        if tag.startswith("M") and is_marker(tok.text):
            cur["marker"] = True
        if tag == "MD":
            cur["part"] = tok.text.lower().startswith("ч")
        if tag == "VD" and cur["part"]:
            cur["VD"].append(value_text(tok.text))
        if tag in ("VA", "VP", "VS"):
            if not tok.text[0].isdigit() and len(tok.text) > 2:
                cur["ordinal"] = True
            cur[tag].append(value_text(tok.text))
            cur["end"] = tok.end
    close()
    return chains


class ChainHMM:
    """Second order HMM tagger. `fit` counts, `predict` runs Viterbi."""

    def __init__(self, k: float = 0.01, lambdas: Tuple[float, float, float] = (0.8, 0.15, 0.05)) -> None:
        self.k = k
        self.lambdas = lambdas
        self.n = len(TAGS)

    def fit(self, data: Sequence[Tuple[List[str], List[str]]]) -> "ChainHMM":
        idx = {t: i for i, t in enumerate(TAGS)}
        n = self.n
        uni = np.zeros(n)
        bi = np.zeros((n + 1, n))  # index n is the sentence start
        tri = np.zeros((n + 1, n + 1, n))
        emit: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n))
        for tokens, tags in data:
            prev2, prev1 = n, n
            for tok, tag in zip(tokens, tags):
                j = idx[tag]
                emit[shape(tok)][j] += 1
                uni[j] += 1
                bi[prev1, j] += 1
                tri[prev2, prev1, j] += 1
                prev2, prev1 = prev1, j
        self._set_counts(uni, bi, tri, dict(emit))
        return self

    def _set_counts(self, uni, bi, tri, emit: Dict[str, np.ndarray]) -> None:
        k, n = self.k, self.n
        self.counts = (uni, bi, tri, emit)
        p_uni = (uni + k) / (uni.sum() + k * n)
        p_bi = (bi + k) / (bi.sum(axis=1, keepdims=True) + k * n)
        p_tri = (tri + k) / (tri.sum(axis=2, keepdims=True) + k * n)
        l3, l2, l1 = self.lambdas
        self.log_trans = np.log(l3 * p_tri + l2 * p_bi[None, :, :] + l1 * p_uni[None, None, :])
        self.emit = emit
        self.emit_totals = sum(emit.values()) if emit else np.zeros(n)
        self.vocab_size = len(emit) + 1
        self._unseen = np.log(k / (self.emit_totals + k * self.vocab_size))
        self._log_emit = {
            s: np.log((c + k) / (self.emit_totals + k * self.vocab_size)) for s, c in emit.items()
        }

    def log_emit(self, token: str) -> np.ndarray:
        return self._log_emit.get(shape(token), self._unseen)

    def predict(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        n = self.n
        trans = self.log_trans
        # state (t_{i-1}, t_i); row n means t_{i-1} is the sentence start
        score = np.full((n + 1, n), -math.inf)
        score[n] = trans[n, n] + self.log_emit(tokens[0])
        backs = []
        for tok in tokens[1:]:
            cand = score[:, :, None] + trans[:, :n, :]  # (t2, t1, t)
            backs.append(cand.argmax(axis=0))
            score = np.full((n + 1, n), -math.inf)
            score[:n] = cand.max(axis=0) + self.log_emit(tok)[None, :]
        t1, t = np.unravel_index(int(np.argmax(score)), score.shape)
        if len(tokens) == 1:
            return [TAGS[int(t)]]
        path = [int(t), int(t1)]
        for back in reversed(backs[1:]):
            path.append(int(back[path[-1], path[-2]]))
        return [TAGS[i] for i in reversed(path)]

    def save(self, path: Path) -> None:
        uni, bi, tri, emit = self.counts
        data = {
            "tags": TAGS,
            "k": self.k,
            "lambdas": list(self.lambdas),
            "uni": uni.tolist(),
            "bi": bi.tolist(),
            "tri": tri.tolist(),
            "emit": {s: c.tolist() for s, c in sorted(emit.items())},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False) + "\n", "utf-8")

    @classmethod
    def load(cls, path: Path) -> "ChainHMM":
        data = json.loads(Path(path).read_text("utf-8"))
        if data["tags"] != TAGS:
            raise ValueError(f"tag set mismatch in {path}")
        model = cls(k=data["k"], lambdas=tuple(data["lambdas"]))
        model._set_counts(
            np.array(data["uni"]),
            np.array(data["bi"]),
            np.array(data["tri"]),
            {s: np.array(c) for s, c in data["emit"].items()},
        )
        return model


class ChainTagger:
    def __init__(self, model: ChainHMM) -> None:
        self.model = model

    @classmethod
    def from_file(cls, path: Path) -> "ChainTagger":
        return cls(ChainHMM.load(path))

    def tagged(self, text: str) -> Tuple[List[Tok], List[str]]:
        # the model is trained on normalized text; normalization keeps offsets
        tokens = tokenize(normalize(text))
        # "ст..161": a repeated dot is noise the model has not seen
        tokens = [t for i, t in enumerate(tokens) if not (t.text == "." and i and tokens[i - 1].text == ".")]
        return tokens, self.model.predict([t.text for t in tokens])

    def find(self, text: str) -> List[Chain]:
        return decode(*self.tagged(text))

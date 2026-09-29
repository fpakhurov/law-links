"""Reference chains as sequence labeling.

Tokens get one of the tags
    O           outside a chain
    MS VS       subpoint marker / value      ("пп.", "подпункт" / "1", "б")
    MP VP       point or part marker / value ("п.", "ч.", "пункта" / "2", "первой")
    MD VD       dropped level (a part above a point in four-level chains)
    MA VA       article marker / value       ("ст.", "статьи" / "3", "19.5")
    X           filler inside a chain        (".", ",", "и", quotes, "в")
and tag sequences are decoded into law_links.grammar.Chain objects, so a
tagger is a drop-in ChainFinder.

Training data is synthetic: chains from a generator covering the phrasing
seen in court decisions, inserted into real sentences from the corpus that
contain no reference markers (their tokens are all O).
"""

import random
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from law_links.grammar import Chain
from law_links.normalize import normalize

TAGS = ["O", "MS", "VS", "MP", "VP", "MD", "VD", "MA", "VA", "X"]
_TOKEN_RE = re.compile(
    r"\d+(?:\.\d+)*(?:\s*-\s*\d+(?:\.\d+)*)?|[А-Яа-яЁёA-Za-z]+|\S"
)
_ORDINALS = {
    "перв": "1", "втор": "2", "трет": "3", "четверт": "4", "пят": "5",
    "шест": "6", "седьм": "7", "восьм": "8", "девят": "9", "десят": "10",
}
_MARKER_RE = re.compile(
    r"(?<![а-яa-z])(?:ст|стать[а-я]*|пп?|пункт[а-я]*|подп|подпункт[а-я]*|ч|част[а-я]*)\.?\s*[\d«\"]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Tok:
    text: str
    start: int
    end: int


def tokenize(text: str) -> List[Tok]:
    return [Tok(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def value_text(token: str) -> str:
    """Normalized value: no spaces in ranges, ordinal words to digits."""
    lower = token.lower()
    for stem, digit in _ORDINALS.items():
        if lower.startswith(stem) and len(lower) > len(stem):
            return digit
    return re.sub(r"\s+", "", lower)


def decode(tokens: Sequence[Tok], tags: Sequence[str]) -> List[Chain]:
    """Group tagged tokens into chains. A chain ends at an O token or when a
    new marker follows an article value ("ст. 15, ст. 64" is two chains)."""
    chains: List[Chain] = []
    cur: Optional[dict] = None

    def close():
        nonlocal cur
        if cur and (cur["VA"] or cur["VP"] or cur["VS"]):
            chains.append(
                Chain(cur["start"], cur["end"], cur["VA"], cur["VP"], cur["VS"])
            )
        cur = None

    for tok, tag in zip(tokens, tags):
        if tag == "O":
            close()
            continue
        if tag.startswith("M") and cur is not None and cur["VA"]:
            close()
        if cur is None:
            if tag == "X":
                continue
            cur = {"start": tok.start, "end": tok.end, "VA": [], "VP": [], "VS": []}
        if tag in ("VA", "VP", "VS"):
            cur[tag].append(value_text(tok.text))
            cur["end"] = tok.end
    close()
    return chains


class TaggerChains:
    """ChainFinder adapter for any tagger with `predict(tokens) -> tags`."""

    def __init__(self, tagger) -> None:
        self.tagger = tagger

    def find(self, text: str) -> List[Chain]:
        tokens = tokenize(text)
        if not tokens:
            return []
        return decode(tokens, self.tagger.predict([t.text for t in tokens]))


# Synthetic data.

_ART = {
    "abbr": ["ст.", "ст", "ст.ст.", "ст. ст.", "Ст."],
    "word": ["статья", "статьи", "статье", "статью", "статьей", "статьёй", "статьями", "статьям", "статей", "Статья", "Статьей"],
}
_PT = {
    "abbr": ["п.", "п", "п.п.", "п.п", "пп.", "Пункт", "ч.", "ч", "ч.ч."],
    "word": ["пункт", "пункта", "пункту", "пунктом", "пункте", "пункты", "пунктами", "пунктов", "Пунктом",
             "часть", "части", "частью", "частями", "частей", "Частью"],
}
_SUB = {
    "abbr": ["пп.", "подп.", "п.п.", "пп"],
    "word": ["подпункт", "подпункта", "подпункту", "подпунктом", "подпунктами", "подпунктов", "Подпунктом"],
}
_PART_ONLY = ["ч.", "ч", "части", "частью", "часть", "частей"]
_POINT_ONLY = ["п.", "п", "пункта", "пунктом", "пункту", "пунктами", "Пунктом", "пункт"]
_ORD = ["первой", "второй", "третьей", "четвертой", "пятой", "первая", "вторая"]
_LETTERS = list("абвгдежзиклмнр")
_LAWS = [
    "УК РФ", "ГК РФ", "НК РФ", "КоАП РФ", "АПК РФ", "ГПК РФ", "ТК РФ", "УПК РФ", "ЖК РФ", "КАС РФ", "КоАП",
    "Гражданского кодекса Российской Федерации", "Налогового кодекса РФ", "Трудового кодекса Российской Федерации",
    "Уголовного кодекса Российской Федерации", "Кодекса РФ об административных правонарушениях",
    "Федерального закона от 26.10.2002 № 127-ФЗ «О несостоятельности (банкротстве)»",
    "Закона № 580-ФЗ", "Закона N 27-ФЗ", "Федерального закона \"О защите прав потребителей\"",
    "Жилищного кодекса РФ", "Семейного кодекса РФ", "Бюджетного кодекса Российской Федерации",
    "названного Кодекса", "настоящего Федерального закона", "Кодекса",
    "Договора", "Правил", "Положения", "постановления Пленума Верховного Суда РФ от 17.03.2004 № 2",
]
_NEGATIVES = [
    "(л.д. {n}-{m})", "(том {k}, л.д. {n})", "т. {k} стр. {n}-{m}", "ст. Отрадная", "ст. Каневская",
    "ком. №№ {n},{m}", "дело № {k}-{n}/2025", "от {d}.{mo}.2024 № {n}", "{n} руб. {m} коп.",
    "в {n} ч. {m} мин.", "п. Заиграево", "г. Самара, ул. Мира, д. {n}", "{n}.{m}.{k}",
]


def _num(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.55:
        return str(rng.randint(1, 400))
    if kind < 0.8:
        return f"{rng.randint(1, 60)}.{rng.randint(1, 12)}"
    if kind < 0.9:
        return f"{rng.randint(1, 20)}.{rng.randint(1, 40)}.{rng.randint(1, 5)}"
    a = rng.randint(1, 300)
    sep = rng.choice(["-", " - ", "- ", " -"])
    return f"{a}{sep}{a + rng.randint(1, 5)}"


def _letter(rng: random.Random) -> List[Tuple[str, str]]:
    letter = rng.choice(_LETTERS)
    quote = rng.random()
    if quote < 0.35:
        return [("«", "X"), (letter, "V"), ("»", "X")]
    if quote < 0.55:
        return [('"', "X"), (letter, "V"), ('"', "X")]
    return [(letter, "V")]


def _values(rng: random.Random, letters: bool, ordinal: bool = False) -> List[Tuple[str, str]]:
    """Value list as (text, role) pieces; role V is refined by the caller."""
    n = rng.choices([1, 2, 3, 4], weights=[70, 18, 8, 4])[0]
    out: List[Tuple[str, str]] = []
    for i in range(n):
        if i:
            sep = rng.choice([",", "и", ",", ", и"]) if i == n - 1 else ","
            out += [(p, "X") for p in sep.replace(",", " , ").split()]
        if letters:
            out += _letter(rng)
        elif ordinal and n == 1:
            out.append((rng.choice(_ORD), "V"))
        else:
            out.append((_num(rng), "V"))
    return out


def _marker(rng: random.Random, forms: dict) -> List[Tuple[str, str]]:
    word = rng.choice(forms["abbr"] if rng.random() < 0.6 else forms["word"])
    pieces = []
    for part in re.findall(r"[А-Яа-яЁё]+|\.", word):
        pieces.append((part, "M" if part != "." else "X"))
    return pieces


def _level(rng, forms, role: str, letters=False, ordinal=False, trailing_dot=False):
    pieces = [(t, r + role if r in "MV" else r) for t, r in _marker(rng, forms)]
    pieces += [(t, r + role if r == "V" else r) for t, r in _values(rng, letters, ordinal)]
    if trailing_dot:
        pieces.append((".", "X"))
    return pieces


def synth_chain(rng: random.Random) -> List[Tuple[str, str]]:
    shape = rng.choices(
        ["a", "pa", "spa", "sppa", "p_in_part", "pp_repeat", "pa_multi", "p_only", "s_only_law"],
        weights=[30, 30, 14, 5, 7, 5, 6, 2, 1],
    )[0]
    trailing = rng.random() < 0.12
    art = lambda: _level(rng, _ART, "A", trailing_dot=trailing)
    if shape == "a":
        return art()
    if shape == "pa":
        return _level(rng, _PT, "P", letters=rng.random() < 0.1, ordinal=rng.random() < 0.1) + _gap(rng) + art()
    if shape == "spa":
        return (_level(rng, _SUB, "S", letters=rng.random() < 0.4) + _gap(rng)
                + _level(rng, _PT, "P") + _gap(rng) + art())
    if shape == "sppa":
        return (_level(rng, _SUB, "S", letters=rng.random() < 0.5) + _gap(rng)
                + _level(rng, {"abbr": _POINT_ONLY, "word": _POINT_ONLY}, "P") + _gap(rng)
                + _level(rng, {"abbr": _PART_ONLY, "word": _PART_ONLY}, "D", ordinal=rng.random() < 0.3)
                + _gap(rng) + art())
    if shape == "p_in_part":
        return (_level(rng, {"abbr": _POINT_ONLY, "word": _POINT_ONLY}, "S", letters=rng.random() < 0.4)
                + _gap(rng)
                + _level(rng, {"abbr": _PART_ONLY, "word": _PART_ONLY}, "P", ordinal=rng.random() < 0.2)
                + _gap(rng) + art())
    if shape == "pp_repeat":
        first = _level(rng, {"abbr": _PART_ONLY, "word": _PART_ONLY}, "P")
        joiner = rng.choice([[(",", "X")], [("и", "X")], [("так", "X"), ("и", "X")]])
        lead = [("как", "X")] if joiner[0][0] == "так" else []
        return lead + first + joiner + _level(rng, {"abbr": _PART_ONLY, "word": _PART_ONLY}, "P") + _gap(rng) + art()
    if shape == "pa_multi":
        a = _level(rng, _PT, "P") + _gap(rng) + _level(rng, _ART, "A")
        return a + [(",", "O")] + synth_chain_simple(rng)
    if shape == "p_only":
        return _level(rng, {"abbr": _POINT_ONLY, "word": _POINT_ONLY}, "P")
    return _level(rng, _SUB, "S") + _gap(rng) + _level(rng, _PT, "P") + _gap(rng) + art()


def synth_chain_simple(rng: random.Random) -> List[Tuple[str, str]]:
    if rng.random() < 0.5:
        return _level(rng, _ART, "A")
    return _level(rng, _PT, "P") + _gap(rng) + _level(rng, _ART, "A")


def _gap(rng: random.Random) -> List[Tuple[str, str]]:
    r = rng.random()
    if r < 0.8:
        return []
    if r < 0.9:
        return [(",", "X")]
    return [("в", "X")]


def _negative(rng: random.Random) -> str:
    return rng.choice(_NEGATIVES).format(
        n=rng.randint(1, 300), m=rng.randint(1, 300), k=rng.randint(1, 12),
        d=rng.randint(1, 28), mo=rng.randint(1, 12),
    )


def plain_sentences(texts: Iterable[str], limit: int) -> List[List[str]]:
    """Corpus sentences without reference markers, as token texts."""
    out: List[List[str]] = []
    for text in texts:
        for line in text.split("\n"):
            for sent in re.split(r"(?<=[.!?])\s+(?=[А-Я])", normalize(line)):
                if 5 <= len(sent) <= 400 and not _MARKER_RE.search(sent):
                    out.append([t.text for t in tokenize(sent)])
                    if len(out) >= limit:
                        return out
    return out


def synth_dataset(
    sentences: List[List[str]], n: int, seed: int = 7, neg_rate: float = 0.3
) -> List[Tuple[List[str], List[str]]]:
    """Insert generated chains (and hard negatives) into real sentences."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        base = list(rng.choice(sentences))
        tokens, tags = [], []
        pos = rng.randint(0, len(base))
        left, right = base[:pos], base[pos:]
        tokens += left
        tags += ["O"] * len(left)
        for _ in range(rng.choices([1, 2], weights=[85, 15])[0]):
            chain = synth_chain(rng)
            for text, tag in chain:
                for tok in tokenize(text):
                    tokens.append(tok.text)
                    tags.append(tag)
            for tok in tokenize(rng.choice(_LAWS)):
                tokens.append(tok.text)
                tags.append("O")
        if rng.random() < neg_rate:
            for tok in tokenize(_negative(rng)):
                tokens.append(tok.text)
                tags.append("O")
        tokens += right
        tags += ["O"] * len(right)
        data.append((tokens, tags))
    return data

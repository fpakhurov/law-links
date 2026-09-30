"""Noise robustness benchmark: dev texts with typing noise, same answers.

Every noise family changes the form of a text but not the references in
it: digits and one-letter values are never touched, so the gold links of
the clean text stay the gold links of the noisy one.

    spaces      "ст. 5" -> "ст.5", "ч. 1 ст." -> "ч.1  ст.", ", " -> ","
    dots        "ст. 5" -> "ст 5", "4.1 " -> "4.1. ", "ст." -> "ст ."
    case        whole text upper or lower case, random words upper case
    homoglyphs  Latin look-alikes inside Cyrillic words ("cт." with Latin c)
    typos       a letter dropped, doubled, swapped or replaced in long words
    glue        the space before a marker dropped ("Руководствуясьст. 23.1")
    breaks      spaces replaced by line breaks and runs of spaces
    all         every family at a lower rate

Usage:
    python -m research.noise [--seeds 5] [--gold tests/gold.json ...]
"""

import argparse
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

from law_links import DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH
from law_links.extractor import Extractor
from scripts.eval import DEFAULT_GOLD, load_cases, score, to_key

_CYR_TO_LAT = {"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
               "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
               "Р": "P", "С": "C", "Т": "T", "Х": "X"}
_MARKER = r"(?:ст|ч|п|пп|подп|абз|стать[а-яё]*|част[а-яё]*|пункт[а-яё]*|подпункт[а-яё]*)"
_MARKER_DOT_RE = re.compile(rf"(?<![А-Яа-яЁё])({_MARKER})\.(\s+)(?=[\dА-Яа-яЁё«\"])", re.IGNORECASE)
_BEFORE_MARKER_RE = re.compile(rf"(?<=[А-Яа-яЁё,])\s+(?={_MARKER}\.?\s*\d)", re.IGNORECASE)
_NUMBER_END_RE = re.compile(r"(?<=\d)(?=[\s,;])")
_WORD_RE = re.compile(r"[А-Яа-яЁё]+")
_LETTERS = "абвгдежзиклмнопрстуфхцчшщыэюя"


def _sub(pattern: re.Pattern, text: str, rng: random.Random, p: float, repl: Callable) -> str:
    return pattern.sub(lambda m: repl(m) if rng.random() < p else m.group(), text)


def spaces(text: str, rng: random.Random, p: float) -> str:
    text = _sub(_MARKER_DOT_RE, text, rng, p, lambda m: m.group(1) + "." + ("" if rng.random() < 0.7 else "   "))
    return _sub(re.compile(r"(?<=[,;]) (?=\S)"), text, rng, p / 2, lambda m: "")


def dots(text: str, rng: random.Random, p: float) -> str:
    text = _sub(_MARKER_DOT_RE, text, rng, p, lambda m: m.group(1) + rng.choice([" ", " . ", "..", "," ]) + m.group(2)[1:])
    return _sub(_NUMBER_END_RE, text, rng, p / 3, lambda m: ".")


def case(text: str, rng: random.Random, p: float) -> str:
    mode = rng.choice(["upper", "lower", "words"])
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    return _sub(_WORD_RE, text, rng, p, lambda m: m.group().upper())


def homoglyphs(text: str, rng: random.Random, p: float) -> str:
    def swap(m: re.Match) -> str:
        word = m.group()
        spots = [i for i, ch in enumerate(word) if ch in _CYR_TO_LAT]
        if len(word) < 2 or not spots:
            return word
        i = rng.choice(spots)
        return word[:i] + _CYR_TO_LAT[word[i]] + word[i + 1 :]

    return _sub(_WORD_RE, text, rng, p, swap)


def typos(text: str, rng: random.Random, p: float) -> str:
    def typo(m: re.Match) -> str:
        word = m.group()
        if len(word) < 5:
            return word
        i = rng.randrange(1, len(word) - 1)
        kind = rng.choice(["drop", "double", "swap", "replace"])
        if kind == "drop":
            return word[:i] + word[i + 1 :]
        if kind == "double":
            return word[:i] + word[i] + word[i:]
        if kind == "swap":
            return word[: i - 1] + word[i] + word[i - 1] + word[i + 1 :]
        letter = rng.choice(_LETTERS)
        return word[:i] + (letter.upper() if word[i].isupper() else letter) + word[i + 1 :]

    return _sub(_WORD_RE, text, rng, p, typo)


def glue(text: str, rng: random.Random, p: float) -> str:
    return _sub(_BEFORE_MARKER_RE, text, rng, p, lambda m: "")


def breaks(text: str, rng: random.Random, p: float) -> str:
    return _sub(re.compile(r" "), text, rng, p / 3, lambda m: rng.choice(["\n", " \n", "  ", "\t", "\n\n  "]))


FAMILIES: Dict[str, Callable[[str, random.Random, float], str]] = {
    "spaces": spaces,
    "dots": dots,
    "case": case,
    "homoglyphs": homoglyphs,
    "typos": typos,
    "glue": glue,
    "breaks": breaks,
}


def perturb(text: str, family: str, rng: random.Random, p: float = 0.5) -> str:
    if family == "all":
        for name in FAMILIES:
            if name != "case" or rng.random() < 0.3:
                text = FAMILIES[name](text, rng, p / 3)
        return text
    return FAMILIES[family](text, rng, p)


def run(extractor, cases: List[Dict], family: str, seeds: int) -> Dict[str, float]:
    tp = fp = fn = 0
    for seed in range(seeds):
        for case_ in cases:
            rng = random.Random(f"{seed}-{case_['id']}-{family}")
            text = case_["text"] if family == "clean" else perturb(case_["text"], family, rng)
            gold = Counter(to_key(link) for link in case_["links"])
            pred = Counter(to_key(l.model_dump()) for l in extractor.extract(text))
            c_tp, c_fp, c_fn = score(gold, pred)
            tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
        if family == "clean":
            break
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, action="append", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--family", action="append", default=None)
    args = parser.parse_args()

    extractor = Extractor.from_files(DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH)
    cases = [c for path in (args.gold or DEFAULT_GOLD) for c in load_cases(path)]
    families = args.family or ["clean", *FAMILIES, "all"]
    print(f"{'family':<12} {'P':>6} {'R':>6} {'F1':>6}")
    for family in families:
        m = run(extractor, cases, family, args.seeds)
        print(f"{family:<12} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")


if __name__ == "__main__":
    main()

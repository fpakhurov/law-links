"""Length-preserving text normalization.

Every replacement maps one character to one character, so offsets in the
normalized text are valid offsets in the original text.
"""

import re

_CHAR_MAP = str.maketrans(
    {
        "«": '"',
        "»": '"',
        "“": '"',
        "”": '"',
        "„": '"',
        "ё": "е",
        "Ё": "Е",
        " ": " ",
        " ": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
        "–": "-",
        "—": "-",
        "‑": "-",
    }
)
_LATIN_TO_CYR = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_HAS_CYR = re.compile(r"[А-Яа-яЁё]")
_HAS_LAT = re.compile(r"[A-Za-z]")


def fold_homoglyphs(word: str) -> str:
    """Replace Latin look-alike letters in mixed-script words ("cт" -> "ст")."""
    if _HAS_CYR.search(word) and _HAS_LAT.search(word):
        return word.translate(_LATIN_TO_CYR)
    return word


def normalize(text: str) -> str:
    result = text.translate(_CHAR_MAP)
    if _HAS_LAT.search(result):
        result = _WORD_RE.sub(lambda m: fold_homoglyphs(m.group()), result)
    assert len(result) == len(text)
    return result

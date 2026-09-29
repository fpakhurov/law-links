"""Length-preserving text normalization.

Every replacement maps one character to one character, so offsets in the
normalized text are valid offsets in the original text.
"""

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
        " ": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
        "–": "-",
        "—": "-",
        "‑": "-",
    }
)


def normalize(text: str) -> str:
    result = text.translate(_CHAR_MAP)
    assert len(result) == len(text)
    return result

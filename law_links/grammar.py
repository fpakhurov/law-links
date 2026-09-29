"""Grammar of reference chains: [subpoint] [point|part] [article].

Chains are written in reverse hierarchy order, e.g. "пп. 1 п. 2 ст. 3".
Each level holds a list of values: "1, 2 и 3", "а, б", "4.6", "43.2-6".
A list is either all numeric or all letters, so prepositions like "в"
are not taken as a letter value after a number.
"""

import re
from dataclasses import dataclass
from typing import List

_LB = r"(?<![0-9a-zа-я])"
_NUM = r"\d+(?:\.\d+)*(?:\s*-\s*\d+(?:\.\d+)*)?"
_LET = r"[а-я](?![0-9a-zа-я])"
_SEP = r"(?:\s*,\s*(?:и\s+)?|\s+и\s+)"
_NUMLIST = rf"{_NUM}(?:{_SEP}{_NUM})*"
_LETLIST = rf"{_LET}(?:{_SEP}{_LET})*"
_ANYLIST = rf"(?:{_NUMLIST}|{_LETLIST})"
_AFTER_MARKER = r"(?:\.\s*|\s+)"

_SUB_MARK = r"(?:подпункт[а-я]*|подп|пп)"
_POINT_MARK = r"(?:пункт[а-я]*|пунт[а-я]*|част[а-я]*|п|ч)"
_ART_MARK = r"(?:стать[а-я]*|ст)"
_GAP = r"\s*,?\s*(?:в\s+)?"

CHAIN_RE = re.compile(
    rf"{_LB}"
    rf"(?:{_SUB_MARK}{_AFTER_MARKER}(?P<sub>{_ANYLIST}){_GAP})?"
    rf"(?:{_LB}{_POINT_MARK}{_AFTER_MARKER}(?P<point>{_ANYLIST}){_GAP})?"
    rf"(?:{_LB}{_ART_MARK}{_AFTER_MARKER}(?P<art>{_NUMLIST}))?",
    re.IGNORECASE,
)
_SEP_RE = re.compile(_SEP, re.IGNORECASE)


@dataclass(frozen=True)
class Chain:
    start: int
    end: int
    articles: List[str]
    points: List[str]
    subpoints: List[str]


def split_values(group: str) -> List[str]:
    if not group:
        return []
    return [re.sub(r"\s+", "", v).lower() for v in _SEP_RE.split(group.strip())]


def find_chains(text: str) -> List[Chain]:
    chains: List[Chain] = []
    for m in CHAIN_RE.finditer(text):
        groups = [g for g in ("sub", "point", "art") if m.group(g)]
        if not groups:
            continue
        chains.append(
            Chain(
                start=m.start(),
                end=max(m.end(g) for g in groups),
                articles=split_values(m.group("art") or ""),
                points=split_values(m.group("point") or ""),
                subpoints=split_values(m.group("sub") or ""),
            )
        )
    return chains

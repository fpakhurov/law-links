"""Law choices for the annotation form: "15 · Налоговый кодекс Российской Федерации".

The display title is the full alias with the law number and quoted name
("Федеральный закон №63-ФЗ «Об адвокатской ...»"), so laws sharing a number
are told apart; codes use their first alias.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from law_links import DEFAULT_ALIASES_PATH

SEPARATOR = " · "


def law_titles(aliases: Dict[str, List[str]]) -> Dict[int, str]:
    titles = {}
    for law_id, names in aliases.items():
        full = [n for n in names if "№" in n and "«" in n and " от " not in n]
        titles[int(law_id)] = (full or names or [f"закон {law_id}"])[0].strip()
    return titles


def load_titles(path: Path = DEFAULT_ALIASES_PATH) -> Dict[int, str]:
    return law_titles(json.loads(Path(path).read_text("utf-8")))


def choice(law_id: int, title: str) -> str:
    return f"{law_id}{SEPARATOR}{title}"


def choices(titles: Dict[int, str]) -> List[str]:
    return [choice(k, v) for k, v in sorted(titles.items())]


def parse_choice(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    head = value.split(SEPARATOR, 1)[0].strip()
    return int(head) if head.isdigit() else None

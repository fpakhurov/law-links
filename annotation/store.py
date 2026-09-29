"""Annotation tasks and labels on disk.

Tasks:  annotation/data/tasks/<batch>.jsonl, one document per line:
        {"doc_id", "text", "source", "mode": "blind" | "suggest"}
Labels: annotation/data/labels/<batch>/<annotator>.jsonl, one record per
        save; the last record of a document wins:
        {"doc_id", "rows", "no_links", "comment", "difficult", "time"}

A row is what the annotator types for one reference:
    {"fragment", "law_id", "law_name", "article", "point", "subpoint"}
Fields may hold comma-separated lists; a row expands into LawLink dicts by
cartesian product, like the service does ("пп. 1, 2 п. 3" -> 2 links).
law_id is None when the law is missing from law_aliases.json (law_name
keeps the name); such rows are kept for dictionary work but give no links.
"""

import itertools
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from law_links import ROOT

DATA_DIR = ROOT / "annotation" / "data"
TASKS_DIR = DATA_DIR / "tasks"
LABELS_DIR = DATA_DIR / "labels"
_NAME_RE = re.compile(r"^[A-Za-zА-Яа-яЁё0-9_.-]{2,40}$")


@dataclass
class Task:
    doc_id: str
    text: str
    source: str
    mode: str = "blind"


@dataclass
class Label:
    doc_id: str
    rows: List[Dict[str, object]] = field(default_factory=list)
    no_links: bool = False
    comment: str = ""
    difficult: bool = False
    time: str = ""


def valid_annotator(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


def batches() -> List[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.jsonl"))


def load_tasks(batch: str) -> List[Task]:
    path = TASKS_DIR / f"{batch}.jsonl"
    return [Task(**json.loads(line)) for line in path.read_text("utf-8").splitlines() if line.strip()]


def write_tasks(batch: str, tasks: List[Task]) -> Path:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    path = TASKS_DIR / f"{batch}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task.__dict__, ensure_ascii=False) + "\n")
    return path


def labels_path(batch: str, annotator: str) -> Path:
    return LABELS_DIR / batch / f"{annotator}.jsonl"


def load_labels(batch: str, annotator: str) -> Dict[str, Label]:
    """Latest label of every document annotated by `annotator`."""
    path = labels_path(batch, annotator)
    if not path.exists():
        return {}
    latest: Dict[str, Label] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["doc_id"]] = Label(**record)
    return latest


def annotators(batch: str) -> List[str]:
    return sorted(p.stem for p in (LABELS_DIR / batch).glob("*.jsonl"))


def save_label(batch: str, annotator: str, label: Label) -> None:
    if not valid_annotator(annotator):
        raise ValueError(f"invalid annotator name: {annotator!r}")
    path = labels_path(batch, annotator)
    path.parent.mkdir(parents=True, exist_ok=True)
    label.time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(label.__dict__, ensure_ascii=False) + "\n")


def split_values(value: Optional[str]) -> List[Optional[str]]:
    """"1, 2" -> ["1", "2"]; empty -> [None]. Spaces inside a value are
    removed ("43.2 - 6" -> "43.2-6"), letters are lower-cased."""
    parts = [re.sub(r"\s+", "", p).lower() for p in (value or "").split(",")]
    parts = [p for p in parts if p]
    return parts or [None]


def expand_row(row: Dict[str, object]) -> List[Dict[str, object]]:
    if row.get("law_id") is None:
        return []
    return [
        {"law_id": int(row["law_id"]), "article": a, "point_article": p, "subpoint_article": s}
        for a, p, s in itertools.product(
            split_values(row.get("article")), split_values(row.get("point")), split_values(row.get("subpoint"))
        )
    ]


def label_links(label: Label) -> List[Dict[str, object]]:
    return [link for row in label.rows for link in expand_row(row)]


def validate_row(text: str, row: Dict[str, object]) -> List[str]:
    """Problems that block adding a row; an empty list means it is fine."""
    problems = []
    fragment = str(row.get("fragment") or "").strip()
    if not fragment:
        problems.append("укажите фрагмент текста со ссылкой")
    elif find_fragment(text, fragment) is None:
        problems.append("фрагмент не найден в тексте, скопируйте его из документа")
    if row.get("law_id") is None and not str(row.get("law_name") or "").strip():
        problems.append("выберите закон или впишите название закона, которого нет в словаре")
    if not any(str(row.get(k) or "").strip() for k in ("article", "point", "subpoint")):
        problems.append("заполните статью, пункт или подпункт")
    return problems


def _fragment_re(fragment: str) -> Optional["re.Pattern"]:
    words = fragment.split()
    return re.compile(r"\s+".join(re.escape(w) for w in words)) if words else None


def find_fragment(text: str, fragment: str) -> Optional[tuple]:
    """(start, end) of the first occurrence, tolerant to whitespace."""
    pattern = _fragment_re(fragment)
    match = pattern.search(text) if pattern else None
    return (match.start(), match.end()) if match else None


def find_all_fragments(text: str, fragment: str) -> List[tuple]:
    pattern = _fragment_re(fragment)
    return [(m.start(), m.end()) for m in pattern.finditer(text)] if pattern else []

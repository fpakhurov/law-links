"""Court decisions from the open dataset TryDotAtwo/russian-legal-ner.

~98k decisions of magistrate courts (2013), administrative and civil cases,
with typed references that are often messy ("ст.12.8  ч. 3",
"Руководствуясьст. 23.1"). The NER spans of the dataset are about roles and
organizations, not references, so only the texts are used: as a second
source of documents for the survey.

Rows are fetched through the dataset viewer API, a sample instead of the
whole parquet file. Whitespace runs (layout padding with no-break spaces)
are collapsed and empty lines dropped; references are not changed. Unlike
sudact.ru, these texts are not always anonymized, and survey fragments are
published: names ("Иванов И.И.", "И.И. Иванов", "Иванов Иван Иванович") are
replaced with "ФИО", and a surname found so is masked in every form.

Usage:
    python -m research.ner_corpus [--n 50] [--split test] [--seed 20261004]
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict

from law_links import ROOT

DATASET = "TryDotAtwo/russian-legal-ner"
NER_DIR = ROOT / "research" / "data" / "ner"
NER_INDEX_PATH = NER_DIR / "index.jsonl"
_ROWS_URL = "https://datasets-server.huggingface.co/rows?"


def fetch_row(split: str, row: int) -> Dict[str, object]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": split, "offset": row, "length": 1}
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(_ROWS_URL + query, timeout=60) as resp:
                return json.load(resp)["rows"][0]["row"]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def clean(text: str) -> str:
    text = text.replace("\r", "").replace("\u00ad", "")  # soft hyphens: "граж\u00adда\u00adнин"
    lines = (re.sub("[ \t\u00a0]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


_UP, _LO = "А-ЯЁ", "а-яё"
_INITIALS = rf"[{_UP}]\.[ ]?[{_UP}]\.?"
_SURNAME = rf"(?:[{_UP}][{_LO}]+(?:-[{_UP}][{_LO}]+)?|[{_UP}]{{3,}}(?:-[{_UP}]{{3,}})?)"  # "Иванов", "ИВАНОВА"
_PATRONYMIC = rf"[{_UP}][{_LO}]+(?:вич|вна|ична|инична)[{_LO}]{{0,3}}"
_NAME_RES = [
    re.compile(rf"\b({_SURNAME})[ ]+[{_UP}][{_LO}]+[ ]+{_PATRONYMIC}\b"),
    re.compile(rf"\b({_SURNAME})[ ]+{_INITIALS}"),
    re.compile(rf"\b{_INITIALS}[ ]?({_SURNAME})\b"),
    re.compile(rf"\b({_SURNAME})[ ]+[{_UP}]\.(?=[ ,;)\n]|$)"),  # "Гузь А."
]
# capitalized words and abbreviations that precede initials but are not surnames;
# names never span a line break ("О.А.\nРассмотрев")
_NOT_SURNAMES = {"РФ", "ООО", "ОАО", "ЗАО", "ПАО", "ИП", "ГИБДД", "ДПС", "УМВД", "МВД", "ОМВД", "АО", "УФССП",
                 "ПОСТАНОВИЛ", "РЕШИЛ", "УСТАНОВИЛ", "ПОСТАНОВЛЕНИЕ", "РЕШЕНИЕ", "Судья", "Мировой",
                 "Председательствующий", "Секретарь", "Истец", "Ответчик", "Гражданин", "Гражданка",
                 "Защитник", "Представитель", "Прокурор", "Инспектор", "Свидетель", "Россия",
                 "Российской", "Федерации", "Федерация", "Кодекса", "Закона", "Республики", "Башкортостан",
                 "Татарстан"}


def anonymize(text: str) -> str:
    stems = set()
    for pattern in _NAME_RES:
        for m in pattern.finditer(text):
            if m.group(1) not in _NOT_SURNAMES:
                stems.add(m.group(1)[: max(4, len(m.group(1)) - 2)])
    for pattern in _NAME_RES:
        text = pattern.sub(lambda m: m.group() if m.group(1) in _NOT_SURNAMES else "ФИО", text)
    for stem in sorted(stems, key=len, reverse=True):
        # a surname is capitalized or all caps; "мировой" is not "Миронова"
        forms = f"{re.escape(stem.capitalize())}[{_LO}]*|{re.escape(stem.upper())}[{_UP}]*"
        text = re.sub(rf"\b(?:{forms})\b", "ФИО", text)
    return text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rows", type=int, default=10_000, help="rows in the split")
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    NER_DIR.mkdir(parents=True, exist_ok=True)
    rows = random.Random(args.seed).sample(range(args.rows), args.n)
    records = []
    for row in rows:
        doc_id = f"ner-{args.split}-{row}"
        raw = clean(fetch_row(args.split, row)["text"])
        (NER_DIR / "raw").mkdir(exist_ok=True)
        (NER_DIR / "raw" / f"{doc_id}.txt").write_text(raw + "\n", "utf-8")  # local only, for checks
        text = anonymize(raw)
        (NER_DIR / f"{doc_id}.txt").write_text(text + "\n", "utf-8")
        url = f"https://huggingface.co/datasets/{DATASET} ({args.split}, строка {row})"
        records.append({"id": doc_id, "kind": "ner", "url": url, "chars": len(text)})
    with open(NER_INDEX_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} documents, {sum(r['chars'] for r in records)} chars -> {NER_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

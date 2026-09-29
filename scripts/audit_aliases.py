"""Audit law_aliases.json content and write a markdown report.

Usage:
    python -m scripts.audit_aliases [--out docs/aliases_audit.md]

Checks:
- id sequence and empty alias lists;
- identical aliases shared by several laws;
- aliases that become identical after lemmatization (resolver-level ambiguity);
- formatting problems: whitespace, unbalanced quotes, editorial notes;
- short or generic aliases that may produce false matches;
- coverage of a checklist of widely cited acts.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from law_links import DEFAULT_ALIASES_PATH, ROOT
from law_links.normalize import normalize
from law_links.resolver import LawResolver

CHECKLIST = {
    "Конституция Российской Федерации": "Конституция РФ",
    "Гражданский кодекс РФ": "ГК РФ",
    "Уголовный кодекс РФ": "УК РФ",
    "Налоговый кодекс РФ": "НК РФ",
    "Трудовой кодекс РФ": "ТК РФ",
    "Семейный кодекс РФ": "СК РФ",
    "Жилищный кодекс РФ": "ЖК РФ",
    "Земельный кодекс РФ": "ЗК РФ",
    "Лесной кодекс РФ": "ЛК РФ",
    "Бюджетный кодекс РФ": "БК РФ",
    "Уголовно-процессуальный кодекс РФ": "УПК РФ",
    "Гражданский процессуальный кодекс РФ": "ГПК РФ",
    "Арбитражный процессуальный кодекс РФ": "АПК РФ",
    "Кодекс административного судопроизводства РФ": "КАС РФ",
    "Кодекс Российской Федерации об административных правонарушениях": "КоАП РФ",
    "Уголовно-исполнительный кодекс РФ": "УИК РФ",
    'Федеральный закон "О банках и банковской деятельности"': None,
    'Федеральный закон "О персональных данных"': None,
    'Закон "О защите прав потребителей"': None,
    'Федеральный закон "Об акционерных обществах"': None,
    'Федеральный закон "Об обществах с ограниченной ответственностью"': None,
    'Федеральный закон "О несостоятельности (банкротстве)"': None,
    'Федеральный закон "О контрактной системе в сфере закупок товаров, работ, услуг для обеспечения государственных и муниципальных нужд"': None,
    'Федеральный закон "Об информации, информационных технологиях и о защите информации"': None,
}


def md_table(header: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
    return "\n".join(lines)


def short(text: str, n: int = 90) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "aliases_audit.md")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    raw: Dict[str, List[str]] = json.loads(args.aliases.read_text("utf-8"))
    resolver = LawResolver(raw)
    by_key: Dict[str, Dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for law_id, key, alias in zip(resolver.alias_law, resolver.alias_keys, (a for v in raw.values() for a in v)):
        by_key[key][law_id].add(normalize(alias).lower())
    n_conflicts = sum(1 for laws in by_key.values() if len(laws) > 1)
    ids = sorted(int(k) for k in raw)
    n_aliases = sum(len(v) for v in raw.values())
    out: List[str] = ["# Аудит law_aliases.json", ""]

    out += ["## Сводка", ""]
    missing_ids = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
    empty = [k for k, v in raw.items() if not v]
    out += [
        f"- законов: {len(raw)}, алиасов: {n_aliases}",
        f"- диапазон id: {ids[0]}..{ids[-1]}, пропуски: {missing_ids or 'нет'}",
        f"- законов без алиасов: {empty or 'нет'}",
        f"- неоднозначных названий после лемматизации: {n_conflicts}",
        "",
    ]

    by_alias: Dict[str, set] = defaultdict(set)
    for law_id, aliases in raw.items():
        for alias in aliases:
            by_alias[normalize(alias).lower().strip()].add(int(law_id))
    dups = sorted(
        ((a, sorted(i)) for a, i in by_alias.items() if len(i) > 1),
        key=lambda x: (-len(x[1]), x[0]),
    )
    out += [f"## Одинаковые алиасы у разных законов: {len(dups)}", ""]
    out += [
        md_table(["алиас", "law_id"], [[short(a), ", ".join(map(str, i))] for a, i in dups[: args.limit]]),
        "",
    ]

    lemma_conflicts = []
    for key, laws in by_key.items():
        aliases = set().union(*laws.values())
        if len(laws) > 1 and len(aliases) > 1:
            lemma_conflicts.append((key, sorted(laws), sorted(aliases)))
    out += [
        f"## Разные алиасы, совпавшие после лемматизации: {len(lemma_conflicts)}",
        "",
        md_table(
            ["ключ лемм", "law_id", "алиасы"],
            [[short(k, 60), ", ".join(map(str, i)), short("; ".join(a), 120)] for k, i, a in lemma_conflicts[: args.limit]],
        ),
        "",
    ]

    fmt_rows = []
    for law_id, aliases in raw.items():
        for alias in aliases:
            problems = []
            if alias != alias.strip() or "  " in alias:
                problems.append("пробелы")
            if alias.count('"') % 2 or alias.count("«") != alias.count("»"):
                problems.append("кавычки")
            if re.search(r"\(в ред\.", alias):
                problems.append("редакция в названии")
            for word in re.findall(r"\w+", alias):
                if re.search(r"[а-яА-ЯёЁ]", word) and re.search(r"[a-zA-Z]", word):
                    problems.append(f"смешение латиницы и кириллицы: {word}")
            if problems:
                fmt_rows.append([law_id, short(alias), ", ".join(problems)])
    out += [f"## Проблемы форматирования: {len(fmt_rows)}", ""]
    out += [md_table(["law_id", "алиас", "проблема"], fmt_rows[: args.limit]), ""]

    generic_rows = []
    for law_id, aliases in raw.items():
        for alias in aliases:
            stripped = alias.strip()
            quoted_only = stripped[:1] in "\"«" and stripped[-1:] in "\"»"
            words = re.findall(r"[А-Яа-яЁё]+", stripped)
            if quoted_only and len(words) <= 2:
                generic_rows.append([law_id, alias])
    out += [
        f"## Короткие названия в кавычках (риск ложных совпадений): {len(generic_rows)}",
        "",
        md_table(["law_id", "алиас"], generic_rows[: args.limit]),
        "",
    ]

    cov_rows = []
    for name, abbr in CHECKLIST.items():
        found = resolver.lookup(name)
        found_abbr = resolver.lookup(abbr) if abbr else []
        cov_rows.append([name, found or "нет", (found_abbr or "нет") if abbr else "-"])
    out += [
        "## Покрытие часто цитируемых актов",
        "",
        "Поиск по полному названию и сокращению с точностью до лемм.",
        "",
        md_table(["акт", "law_id по названию", "law_id по сокращению"], cov_rows),
        "",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out), encoding="utf-8")
    print(f"Report written: {args.out}")


if __name__ == "__main__":
    main()

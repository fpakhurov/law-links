"""Gradio app for annotating legal references.

Run:
    python -m annotation.app [--host 127.0.0.1] [--port 7860]

Guidelines: docs/ANNOTATION.md. Tasks come from annotation/make_tasks.py,
labels are appended to annotation/data/labels/<batch>/<annotator>.jsonl.
"""

import argparse
import html
import sys
from collections import Counter
from functools import lru_cache
from typing import Dict, List

import gradio as gr

from annotation.laws import choice, choices, load_titles, parse_choice
from annotation.store import (
    Label,
    batches,
    expand_row,
    find_all_fragments,
    load_labels,
    load_tasks,
    save_label,
    valid_annotator,
    validate_row,
)
from law_links import ROOT

TITLES = load_titles()
LAW_CHOICES = choices(TITLES)
TABLE_HEADERS = ["#", "фрагмент", "закон", "статья", "пункт/часть", "подпункт", "ссылок"]
GUIDELINES = (ROOT / "docs" / "ANNOTATION.md").read_text("utf-8")


@lru_cache(maxsize=1)
def extractor():
    from law_links import DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH
    from law_links.extractor import Extractor

    return Extractor.from_files(DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH)


def suggestions(text: str) -> List[Dict[str, object]]:
    from annotation.suggest import suggestion_rows

    return suggestion_rows(extractor(), text)


def law_label(row: Dict[str, object]) -> str:
    if row.get("law_id") is None:
        return f"нет в словаре: {row.get('law_name')}"
    return choice(int(row["law_id"]), TITLES.get(int(row["law_id"]), ""))


def table(rows: List[Dict[str, object]]) -> List[List[object]]:
    return [
        [i + 1, r["fragment"], law_label(r), r.get("article") or "", r.get("point") or "",
         r.get("subpoint") or "", len(expand_row(r))]
        for i, r in enumerate(rows)
    ]


_MARK_STYLE = {
    "размечено": "background: rgba(46, 160, 67, 0.35); border-bottom: 2px solid rgb(46, 160, 67);",
    "подсказка": "background: rgba(219, 124, 38, 0.30); border-bottom: 2px dashed rgb(219, 124, 38);",
}
_DOC_STYLE = "white-space: pre-wrap; line-height: 1.7; max-height: 70vh; overflow-y: auto; padding: 8px;"


def highlighted(text: str, rows: List[Dict[str, object]], suggested: List[Dict[str, object]]) -> str:
    """Document as HTML with annotated and suggested fragments marked inline."""
    spans = []
    for label, items in (("размечено", rows), ("подсказка", suggested)):
        for fragment in {str(r["fragment"]) for r in items}:
            spans.extend((start, end, label) for start, end in find_all_fragments(text, fragment))
    spans.sort(key=lambda s: (s[0], s[2] != "размечено"))
    parts, pos = [], 0
    for start, end, label in spans:
        if start < pos:
            continue
        parts.append(html.escape(text[pos:start]))
        parts.append(f'<mark title="{label}" style="{_MARK_STYLE[label]} color: inherit;">{html.escape(text[start:end])}</mark>')
        pos = end
    parts.append(html.escape(text[pos:]))
    legend = " ".join(
        f'<mark style="{style} color: inherit;">{label}</mark>' for label, style in _MARK_STYLE.items()
    )
    return f'<div style="margin-bottom: 6px;">{legend}</div><div style="{_DOC_STYLE}">{"".join(parts)}</div>'


def view(state: Dict[str, object]):
    """Values for every component that shows the current document."""
    if not state.get("tasks"):
        return ("Выберите пакет и нажмите «Начать».", "", [], False, False, "", gr.update(visible=False))
    task = state["tasks"][state["idx"]]
    done = state["done"]
    suggested = state.get("suggested", []) if task["mode"] == "suggest" else []
    status = (
        f"**{state['batch']}**, документ {state['idx'] + 1} из {len(state['tasks'])}, "
        f"размечено вами: {len(done)}. Режим: `{task['mode']}`. "
        f"{'Сохранён ранее.' if task['doc_id'] in done else 'Ещё не сохранён.'}  \n"
        f"Источник: {task['source']}"
    )
    return (
        status,
        highlighted(task["text"], state["rows"], suggested),
        table(state["rows"]),
        state["no_links"],
        state["difficult"],
        state["comment"],
        gr.update(visible=task["mode"] == "suggest"),
    )


def open_doc(state: Dict[str, object], idx: int) -> Dict[str, object]:
    state["idx"] = idx
    task = state["tasks"][idx]
    saved = load_labels(state["batch"], state["annotator"]).get(task["doc_id"])
    state["done"] = set(load_labels(state["batch"], state["annotator"]))
    state["rows"] = [dict(r) for r in saved.rows] if saved else []
    state["no_links"] = saved.no_links if saved else False
    state["difficult"] = saved.difficult if saved else False
    state["comment"] = saved.comment if saved else ""
    state["suggested"] = suggestions(task["text"]) if task["mode"] == "suggest" else []
    return state


def start(annotator: str, batch: str, state: Dict[str, object]):
    annotator = (annotator or "").strip()
    if not valid_annotator(annotator):
        gr.Warning("Имя: 2-40 букв, цифр, '_', '-' или '.', без пробелов")
        return (state, *view(state))
    if not batch:
        gr.Warning("Выберите пакет")
        return (state, *view(state))
    tasks = [t.__dict__ for t in load_tasks(batch)]
    done = set(load_labels(batch, annotator))
    first = next((i for i, t in enumerate(tasks) if t["doc_id"] not in done), 0)
    state = {"annotator": annotator, "batch": batch, "tasks": tasks}
    state = open_doc(state, first)
    return (state, *view(state))


def add_row(state, fragment, law, law_name, article, point, subpoint):
    if not state.get("tasks"):
        gr.Warning("Сначала нажмите «Начать»")
        return (state, *view(state), fragment, law, law_name, article, point, subpoint)
    row = {
        "fragment": (fragment or "").strip(),
        "law_id": parse_choice(law),
        "law_name": (law_name or "").strip() if parse_choice(law) is None else "",
        "article": (article or "").strip(),
        "point": (point or "").strip(),
        "subpoint": (subpoint or "").strip(),
    }
    problems = validate_row(state["tasks"][state["idx"]]["text"], row)
    if problems:
        gr.Warning("; ".join(problems))
        return (state, *view(state), fragment, law, law_name, article, point, subpoint)
    state["rows"].append(row)
    state["no_links"] = False
    return (state, *view(state), "", None, "", "", "", "")


def delete_row(state, number):
    rows = state.get("rows", [])
    if number is None or not 1 <= int(number) <= len(rows):
        gr.Warning(f"Номер строки от 1 до {len(rows)}")
    else:
        rows.pop(int(number) - 1)
    return (state, *view(state))


def _row_key(r: Dict[str, object]) -> tuple:
    return (r["fragment"], r["law_id"], r["article"], r["point"], r["subpoint"])


def accept_suggestions(state):
    """Add suggested rows that are not in the table yet. Counts matter: a
    reference repeated in the text is suggested, and kept, several times."""
    have = Counter(_row_key(r) for r in state["rows"])
    for r in state.get("suggested", []):
        if have[_row_key(r)] > 0:
            have[_row_key(r)] -= 1
        else:
            state["rows"].append(dict(r))
    return (state, *view(state))


def save_and_next(state, no_links, difficult, comment):
    if not state.get("tasks"):
        return (state, *view(state))
    if not state["rows"] and not no_links:
        gr.Warning("Добавьте ссылки или отметьте «В документе нет ссылок»")
        return (state, *view(state))
    if state["rows"] and no_links:
        gr.Warning("Отмечено «нет ссылок», но строки есть: удалите строки или снимите отметку")
        return (state, *view(state))
    task = state["tasks"][state["idx"]]
    save_label(state["batch"], state["annotator"], Label(
        doc_id=task["doc_id"], rows=state["rows"], no_links=bool(no_links),
        comment=(comment or "").strip(), difficult=bool(difficult),
    ))
    gr.Info(f"Сохранено: {len(state['rows'])} строк")
    nxt = min(state["idx"] + 1, len(state["tasks"]) - 1)
    state = open_doc(state, nxt)
    return (state, *view(state))


def move(state, step):
    if state.get("tasks"):
        state = open_doc(state, max(0, min(len(state["tasks"]) - 1, state["idx"] + step)))
    return (state, *view(state))


def build() -> gr.Blocks:
    with gr.Blocks(title="Разметка юридических ссылок") as app:
        state = gr.State({})
        gr.Markdown("## Разметка юридических ссылок")
        with gr.Row():
            annotator = gr.Textbox(label="Разметчик", placeholder="ivan", scale=2)
            batch = gr.Dropdown(label="Пакет", choices=batches(), value=None, scale=2)
            start_btn = gr.Button("Начать", variant="primary", scale=1)
        status = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=3):
                doc = gr.HTML(label="Документ")
            with gr.Column(scale=2):
                fragment = gr.Textbox(label="Фрагмент со ссылкой (скопируйте из текста)", lines=2)
                law = gr.Dropdown(label="Закон", choices=LAW_CHOICES, filterable=True, value=None)
                law_name = gr.Textbox(label="Закона нет в словаре: название")
                with gr.Row():
                    article = gr.Textbox(label="Статья")
                    point = gr.Textbox(label="Пункт / часть")
                    subpoint = gr.Textbox(label="Подпункт")
                add_btn = gr.Button("Добавить ссылку", variant="primary")
                accept_btn = gr.Button("Принять подсказки", visible=False)
        rows = gr.Dataframe(headers=TABLE_HEADERS, label="Ссылки документа", interactive=False, wrap=True)
        with gr.Row():
            delete_no = gr.Number(label="Удалить строку №", precision=0, scale=1)
            delete_btn = gr.Button("Удалить", scale=1)
            no_links = gr.Checkbox(label="В документе нет ссылок", scale=1)
            difficult = gr.Checkbox(label="Сложный документ", scale=1)
        comment = gr.Textbox(label="Комментарий (сомнения, спорные места)")
        with gr.Row():
            prev_btn = gr.Button("← Назад")
            save_btn = gr.Button("Сохранить и далее", variant="primary")
            next_btn = gr.Button("Пропустить →")
        with gr.Accordion("Правила разметки", open=False):
            gr.Markdown(GUIDELINES)

        shown = [status, doc, rows, no_links, difficult, comment, accept_btn]
        form = [fragment, law, law_name, article, point, subpoint]
        start_btn.click(start, [annotator, batch, state], [state, *shown])
        add_btn.click(add_row, [state, *form], [state, *shown, *form])
        delete_btn.click(delete_row, [state, delete_no], [state, *shown])
        accept_btn.click(accept_suggestions, [state], [state, *shown])
        save_btn.click(save_and_next, [state, no_links, difficult, comment], [state, *shown])
        prev_btn.click(lambda s: move(s, -1), [state], [state, *shown])
        next_btn.click(lambda s: move(s, 1), [state], [state, *shown])
    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build().launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

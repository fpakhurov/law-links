"""Survey for checking found legal references (Hugging Face Space).

Shows prepared items (items.jsonl, built by `python -m annotation.survey`)
and stores yes/no answers. Every voter gets the items with the fewest
votes that they have not answered yet, with a control item (known answer)
every CONTROL_EVERY answers.

Answers go to VOTES_DIR/votes-<process id>.jsonl. With HF_TOKEN and
VOTES_REPO set (Space secrets), the folder is pushed to that dataset repo
every few minutes, and votes already there are loaded at startup.
"""

import html
import json
import os
import random
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr

HERE = Path(__file__).resolve().parent
ITEMS_PATH = Path(os.getenv("ITEMS_PATH", HERE / "items.jsonl"))
VOTES_DIR = Path(os.getenv("VOTES_DIR", HERE / "votes"))
VOTES_REPO = os.getenv("VOTES_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TARGET_VOTES = 2
CONTROL_EVERY = 8

ANSWERS = {
    "verify": [("yes", "Верно"), ("no_law", "Не тот закон"), ("no_numbers", "Не те номера"),
               ("no_ref", "Это не ссылка"), ("unsure", "Не понять")],
    "missed": [("yes", "Да, это ссылка на норму"), ("no", "Нет"), ("unsure", "Не понять")],
}
INTRO = """
## Проверка найденных юридических ссылок

Программа ищет в судебных решениях ссылки на нормы законов: `ч. 3 ст. 158 УК РФ` -> статья 158, часть 3, Уголовный кодекс. Помогите оценить, насколько верно она это делает.

Вопросов два вида. **Верно ли найдено**: жёлтым выделен фрагмент, ниже то, как программа его поняла. «Верно», если совпадают закон и все номера (статья, часть или пункт, подпункт). Пункт внутри части (`п. 6 ч. 1 ст. 24.5`) записывается как «часть/пункт 1, подпункт 6», абзацы и примечания не учитываются. **Есть ли здесь ссылка**: «да», если выделенный фрагмент указывает на статью, часть или пункт закона или кодекса; пункты договоров, правил, приказов, листы дела (`л.д. 55`) и время (`10 ч. 30 мин.`) ссылками не считаются.

Один ответ - несколько секунд, отвечайте сколько удобно. Ответы анонимны и публикуются в открытом датасете для всех: сохраняются только случайный номер браузера, вопрос, ответ и время ответа.
"""


def load_items() -> Dict[str, Dict[str, object]]:
    items = {}
    for line in ITEMS_PATH.read_text("utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["item_id"]] = item
    return items


ITEMS = load_items()
REGULAR = [i for i, it in ITEMS.items() if it["kind"] != "control"]
CONTROLS = [i for i, it in ITEMS.items() if it["kind"] == "control"]

_lock = threading.Lock()
_votes_per_item: Counter = Counter()
_answered: Dict[str, set] = defaultdict(set)
_scheduler = None
_votes_file = VOTES_DIR / "current" / f"votes-{uuid.uuid4().hex[:12]}.jsonl"


def _remember(vote: Dict[str, object]) -> None:
    _answered[vote["voter"]].add(vote["item_id"])
    if vote["item_id"] in ITEMS and ITEMS[vote["item_id"]]["kind"] != "control":
        _votes_per_item[vote["item_id"]] += 1


def _load_existing() -> None:
    """Votes stored before this process started (dataset repo or local)."""
    folders = [VOTES_DIR]
    if VOTES_REPO and HF_TOKEN:
        from huggingface_hub import snapshot_download

        try:
            folders.append(Path(snapshot_download(
                VOTES_REPO, repo_type="dataset", token=HF_TOKEN, allow_patterns=["votes/*.jsonl"],
                local_dir=VOTES_DIR / "remote",
            )))
        except Exception as exc:  # a new, empty dataset has nothing to download
            print(f"no remote votes loaded: {exc}")
    seen = set()
    for folder in folders:
        for path in folder.rglob("votes-*.jsonl"):
            for line in path.read_text("utf-8").splitlines():
                if line.strip():
                    vote = json.loads(line)
                    if vote["vote_id"] not in seen:
                        seen.add(vote["vote_id"])
                        _remember(vote)
    print(f"loaded {len(seen)} votes")


def _start_scheduler() -> None:
    global _scheduler
    _votes_file.parent.mkdir(parents=True, exist_ok=True)
    if VOTES_REPO and HF_TOKEN:
        from huggingface_hub import CommitScheduler

        _scheduler = CommitScheduler(
            repo_id=VOTES_REPO, repo_type="dataset", folder_path=_votes_file.parent,
            path_in_repo="votes", every=2, token=HF_TOKEN, private=False,
        )


def save_vote(vote: Dict[str, object]) -> None:
    line = json.dumps(vote, ensure_ascii=False) + "\n"
    lock = _scheduler.lock if _scheduler else _lock
    with lock:
        with open(_votes_file, "a", encoding="utf-8") as f:
            f.write(line)
    with _lock:
        _remember(vote)


def next_item(voter: str) -> Optional[str]:
    with _lock:
        done = _answered[voter]
        n_done = len(done)
        if n_done % CONTROL_EVERY == CONTROL_EVERY // 2:
            open_controls = [c for c in CONTROLS if c not in done]
            if open_controls:
                return random.choice(open_controls)
        left = [i for i in REGULAR if i not in done]
        if not left:
            return None
        fewest = min(_votes_per_item[i] for i in left)
        return random.choice([i for i in left if _votes_per_item[i] == fewest])


def card(item_id: Optional[str]) -> str:
    if item_id is None:
        return "<p><b>Вопросы закончились. Спасибо!</b></p>"
    it = ITEMS[item_id]
    text = (
        html.escape(it["before"]).replace("\n", "<br>")
        + '<mark style="background: rgba(250, 204, 21, 0.45); color: inherit; padding: 1px 2px;">'
        + html.escape(it["fragment"]) + "</mark>"
        + html.escape(it["after"]).replace("\n", "<br>")
    )
    if it["kind"] == "missed":
        question = "Есть ли в выделенном фрагменте ссылка на статью, часть или пункт закона?"
        reading = ""
    else:
        question = "Программа поняла выделенный фрагмент так. Верно?"
        reading = (
            '<div style="margin: 10px 0; padding: 8px 10px; border-left: 3px solid rgb(250, 204, 21);">'
            f"<b>{html.escape(it['claim'])}</b></div>"
        )
    return (
        f'<div style="line-height: 1.7; font-size: 1.05em;">{text}</div>'
        f'<p style="margin-top: 14px;"><b>{question}</b></p>{reading}'
    )


def show(voter_state: Dict[str, object]):
    item_id = next_item(voter_state["voter"])
    voter_state["item"] = item_id
    voter_state["shown_at"] = time.time()
    kind = ITEMS[item_id]["kind"] if item_id else None
    verify_visible = kind in ("verify", "control")
    missed_visible = kind == "missed"
    count = len(_answered[voter_state["voter"]])
    return (
        voter_state,
        card(item_id),
        f"Вы ответили: {count}",
        gr.update(visible=verify_visible),
        gr.update(visible=missed_visible),
    )


def start(stored: Optional[Dict[str, object]]):
    voter = (stored or {}).get("voter") or uuid.uuid4().hex
    state = {"voter": voter, "item": None, "shown_at": time.time()}
    return (*show(state), {"voter": voter})


def answer(state: Dict[str, object], value: str):
    item_id = state.get("item")
    if item_id:
        save_vote({
            "vote_id": uuid.uuid4().hex,
            "item_id": item_id,
            "kind": ITEMS[item_id]["kind"],
            "answer": value,
            "voter": state["voter"],
            "ms": int(1000 * (time.time() - state.get("shown_at", time.time()))),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return show(state)


def build() -> gr.Blocks:
    with gr.Blocks(title="Проверка юридических ссылок") as app:
        stored = gr.BrowserState({}, storage_key="law-links-survey")
        state = gr.State({})
        gr.Markdown(INTRO)
        progress = gr.Markdown()
        item_html = gr.HTML()
        with gr.Row(visible=False) as verify_row:
            verify_buttons = [(gr.Button(label, variant="primary" if v == "yes" else "secondary"), v)
                              for v, label in ANSWERS["verify"]]
        with gr.Row(visible=False) as missed_row:
            missed_buttons = [(gr.Button(label, variant="primary" if v == "yes" else "secondary"), v)
                              for v, label in ANSWERS["missed"]]
        outputs = [state, item_html, progress, verify_row, missed_row]
        for button, value in verify_buttons + missed_buttons:
            button.click(lambda s, v=value: answer(s, v), [state], outputs)
        app.load(start, [stored], [*outputs, stored])
    return app


_load_existing()
_start_scheduler()
demo = build()

if __name__ == "__main__":
    # A Space (SPACE_ID is set) serves 0.0.0.0:7860; locally 127.0.0.1:7861
    on_space = bool(os.getenv("SPACE_ID"))
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0" if on_space else "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860" if on_space else "7861")),
    )

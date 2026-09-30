"""Survey for checking found legal references (Hugging Face Space).

Shows prepared items (items.jsonl, built by `python -m annotation.survey`)
and stores yes/no answers. A voter picks a nickname on the first visit (kept
in browser storage with a random voter id). Every voter gets the items with
the fewest votes that they have not answered yet, with a control item
(known answer) every CONTROL_EVERY answers. A leaderboard shows nicknames
by the number of answers.

Answers go to VOTES_DIR/votes-<process id>.jsonl. With HF_TOKEN and
VOTES_REPO set (Space secret and variable), the folder is pushed to that
public dataset repo every few minutes, and votes already there are loaded
at startup.
"""

import html
import json
import os
import random
import re
import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr

try:
    import spaces

    # The Space runs on ZeroGPU hardware, which refuses to start without a
    # declared GPU function. The survey needs no GPU; this is never called.
    @spaces.GPU
    def _zero_gpu_placeholder() -> None:
        return None

except ImportError:  # outside Hugging Face Spaces
    pass

HERE = Path(__file__).resolve().parent
ITEMS_PATH = Path(os.getenv("ITEMS_PATH", HERE / "items.jsonl"))
VOTES_DIR = Path(os.getenv("VOTES_DIR", HERE / "votes"))
VOTES_REPO = os.getenv("VOTES_REPO", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
CONTROL_EVERY = 8
LEADERBOARD_SIZE = 10
_NAME_RE = re.compile(r"^[\w .\-]{2,24}$")

ANSWERS = {
    "verify": [("yes", "Верно"), ("no_law", "Не тот закон"), ("no_numbers", "Не те номера"),
               ("no_ref", "Это не ссылка"), ("unsure", "Не понять")],
    "missed": [("yes", "Да, закон или кодекс"), ("other_doc", "Да, но другой документ"),
               ("no", "Нет"), ("unsure", "Не понять")],
}
INFO = """
### Что мы проверяем

Программа ищет в судебных решениях ссылки на статьи законов. Жёлтым выделено то, что она нашла или пропустила.

**Верно ли понято.** Под текстом номера в том же порядке, что в тексте, и закон: `п. 6 ч. 1 ст. 24.5 КоАП РФ` -> `6 · 1 · ст. 24.5 — Кодекс об административных правонарушениях`. «Верно», если закон тот же и все номера совпадают.

**Есть ли ссылка.** «Да, закон или кодекс» - ссылка на статью, часть или пункт закона. «Да, но другой документ» - пункт договора, правил, приказа, постановления. «Нет» - это вообще не ссылка (лист дела, время, номер дома).

Сомневаетесь - жмите «Не понять». Среди вопросов есть проверочные с известным ответом.

Ответы публикуются в открытом датасете [fpakhurov/law-links-votes](https://huggingface.co/datasets/fpakhurov/law-links-votes): имя, случайный номер браузера, вопрос, ответ, время.
"""
NAME_PROMPT = """
### Придумайте имя

Оно будет видно в рейтинге и попадёт в открытый датасет вместе с ответами, поэтому лучше выдуманное. 2-24 символа: буквы, цифры, пробел, `_ . -`.
"""
CSS = """
.gradio-container { max-width: 760px !important; margin: 0 auto; }
#topbar { align-items: center; flex-wrap: nowrap; gap: 8px; }
#topbar button { flex: 0 0 auto; min-width: 44px; }
#title { flex: 1 1 auto; text-align: center; font-weight: 600; font-size: 1.05em; }
#item { font-size: 1.05em; line-height: 1.7; }
.answers { flex-wrap: wrap; gap: 8px; }
.answers button { flex: 1 1 140px; min-height: 48px; }
.overlay { position: fixed !important; inset: 0; z-index: 1000; background: rgba(0, 0, 0, 0.55);
           display: flex; align-items: center; justify-content: center; padding: 16px; }
.overlay > .overlay-card { width: 100%; max-width: 480px; max-height: 85vh; overflow-y: auto;
                           flex: 0 0 auto !important; height: auto !important;
                           background: var(--background-fill-primary); border-radius: 12px; padding: 16px; }
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
_names: Dict[str, str] = {}  # voter -> nickname
_owners: Dict[str, str] = {}  # nickname in lower case -> voter
_scheduler = None
_votes_file = VOTES_DIR / "current" / f"votes-{uuid.uuid4().hex[:12]}.jsonl"


def _remember(vote: Dict[str, object]) -> None:
    _answered[vote["voter"]].add(vote["item_id"])
    if vote.get("name"):
        _names[vote["voter"]] = vote["name"]
        _owners.setdefault(vote["name"].lower(), vote["voter"])
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


def claim_name(voter: str, name: str) -> Tuple[bool, str]:
    """Register a nickname for the voter; (ok, normalized name or error)."""
    name = " ".join((name or "").split())
    if not _NAME_RE.match(name):
        return False, "Имя: 2-24 символа, буквы, цифры, пробел, _ . -"
    with _lock:
        owner = _owners.get(name.lower())
        if owner and owner != voter:
            return False, "Это имя уже занято, придумайте другое"
        old = _names.get(voter)
        if old and old.lower() != name.lower() and _owners.get(old.lower()) == voter:
            del _owners[old.lower()]
        _owners[name.lower()] = voter
        _names[voter] = name
    return True, name


def next_item(voter: str) -> Optional[str]:
    with _lock:
        done = _answered[voter]
        if len(done) % CONTROL_EVERY == CONTROL_EVERY // 2:
            open_controls = [c for c in CONTROLS if c not in done]
            if open_controls:
                return random.choice(open_controls)
        left = [i for i in REGULAR if i not in done]
        if not left:
            return None
        fewest = min(_votes_per_item[i] for i in left)
        return random.choice([i for i in left if _votes_per_item[i] == fewest])


def leaderboard(voter: Optional[str]) -> str:
    with _lock:
        counts = sorted(
            ((len(done), _names.get(v, "без имени"), v) for v, done in _answered.items() if done),
            key=lambda x: (-x[0], x[1].lower()),
        )
    rows = []
    for place, (count, name, v) in enumerate(counts[:LEADERBOARD_SIZE], 1):
        me = " (вы)" if v == voter else ""
        rows.append(f"<tr><td>{place}</td><td>{html.escape(name)}{me}</td><td>{count}</td></tr>")
    mine = next(((p, c) for p, (c, _, v) in enumerate(counts, 1) if v == voter), None)
    footer = f"<p>Ваше место: {mine[0]}, ответов: {mine[1]}</p>" if mine else "<p>Вы ещё не отвечали.</p>"
    if not rows:
        return "<h3>Рейтинг</h3><p>Пока никто не ответил.</p>"
    return (
        "<h3>Рейтинг</h3><table style='width:100%'><tr><th>#</th><th>Имя</th><th>Ответов</th></tr>"
        + "".join(rows) + "</table>" + footer
    )


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
        question = "Программа не посчитала это ссылкой. Это ссылка?"
        reading = ""
    else:
        question = "Программа поняла выделенное так. Закон и все номера совпадают?"
        reading = (
            '<div style="margin: 10px 0; padding: 8px 10px; border-left: 3px solid rgb(250, 204, 21);">'
            f"<b>{html.escape(it['claim'])}</b></div>"
        )
    return f"<div>{text}</div><p style='margin-top: 14px;'><b>{question}</b></p>{reading}"


def show(state: Dict[str, object]):
    """Next item for the voter: state, card, progress, answer rows."""
    item_id = next_item(state["voter"])
    state["item"] = item_id
    state["shown_at"] = time.time()
    kind = ITEMS[item_id]["kind"] if item_id else None
    count = len(_answered[state["voter"]])
    return (
        state,
        card(item_id),
        f"{html.escape(state.get('name', ''))}, ваших ответов: {count}",
        gr.update(visible=kind in ("verify", "control")),
        gr.update(visible=kind == "missed"),
    )


def start(stored: Optional[Dict[str, object]]):
    voter = (stored or {}).get("voter") or uuid.uuid4().hex
    name = (stored or {}).get("name") or ""
    state = {"voter": voter, "name": name, "item": None, "shown_at": time.time()}
    if name and claim_name(voter, name)[0]:
        return (*show(state), gr.update(visible=False), "", {"voter": voter, "name": name})
    return (state, "", "", gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=True), "", {"voter": voter, "name": ""})


def set_name(state: Dict[str, object], name: str):
    ok, result = claim_name(state["voter"], name)
    if not ok:
        return (state, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.update(visible=True),
                f"<p style='color: #d9534f;'>{html.escape(result)}</p>", gr.skip())
    state["name"] = result
    return (*show(state), gr.update(visible=False), "", {"voter": state["voter"], "name": result})


def answer(state: Dict[str, object], value: str):
    item_id = state.get("item")
    if item_id and state.get("name"):
        save_vote({
            "vote_id": uuid.uuid4().hex,
            "item_id": item_id,
            "kind": ITEMS[item_id]["kind"],
            "answer": value,
            "voter": state["voter"],
            "name": state["name"],
            "ms": int(1000 * (time.time() - state.get("shown_at", time.time()))),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
    return show(state)


def build() -> gr.Blocks:
    with gr.Blocks(title="Проверка юридических ссылок") as app:
        # A fixed key, so the voter id and nickname survive restarts of the app
        # (the default key is random per process). It only guards against
        # accidental corruption; the stored values are not sensitive.
        stored = gr.BrowserState(
            {}, storage_key="law-links-survey",
            secret=os.getenv("BROWSER_STATE_SECRET", "law-links-survey-v1"),
        )
        state = gr.State({})
        with gr.Row(elem_id="topbar"):
            board_btn = gr.Button("Рейтинг", size="sm", scale=0, min_width=44)
            gr.HTML("<div>Проверка юридических ссылок</div>", elem_id="title")
            info_btn = gr.Button("i", size="sm", scale=0, min_width=44)
        progress = gr.Markdown()
        item_html = gr.HTML(elem_id="item")
        with gr.Row(visible=False, elem_classes=["answers"]) as verify_row:
            verify_buttons = [(gr.Button(label, variant="primary" if v == "yes" else "secondary"), v)
                              for v, label in ANSWERS["verify"]]
        with gr.Row(visible=False, elem_classes=["answers"]) as missed_row:
            missed_buttons = [(gr.Button(label, variant="primary" if v == "yes" else "secondary"), v)
                              for v, label in ANSWERS["missed"]]

        with gr.Column(visible=False, elem_classes=["overlay"]) as name_modal:
            with gr.Column(elem_classes=["overlay-card"]):
                gr.Markdown(NAME_PROMPT)
                name_box = gr.Textbox(show_label=False, placeholder="Например, Сова-42", max_lines=1)
                name_error = gr.HTML()
                name_btn = gr.Button("Начать", variant="primary")
        with gr.Column(visible=False, elem_classes=["overlay"]) as info_modal:
            with gr.Column(elem_classes=["overlay-card"]):
                gr.Markdown(INFO)
                info_close = gr.Button("Понятно", variant="primary")
        with gr.Column(visible=False, elem_classes=["overlay"]) as board_modal:
            with gr.Column(elem_classes=["overlay-card"]):
                board_html = gr.HTML()
                board_close = gr.Button("Закрыть", variant="primary")

        shown = [state, item_html, progress, verify_row, missed_row]
        for button, value in verify_buttons + missed_buttons:
            button.click(lambda s, v=value: answer(s, v), [state], shown)
        for trigger in (name_btn.click, name_box.submit):
            trigger(set_name, [state, name_box], [*shown, name_modal, name_error, stored])
        info_btn.click(lambda: gr.update(visible=True), None, info_modal)
        info_close.click(lambda: gr.update(visible=False), None, info_modal)
        board_btn.click(lambda s: (leaderboard(s.get("voter")), gr.update(visible=True)), [state],
                        [board_html, board_modal])
        board_close.click(lambda: gr.update(visible=False), None, board_modal)
        app.load(start, [stored], [*shown, name_modal, name_error, stored])
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
        css=CSS,
    )

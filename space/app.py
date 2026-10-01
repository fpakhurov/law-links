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
N_SCREEN = 12  # outputs of screen() without the joke overlay pair
# A joke from the anekdot.ru informer for websites every JOKE_GAP answers
JOKE_GAP = tuple(int(x) for x in os.getenv("JOKE_GAP", "12,20").split(","))
JOKE_INFORMER = "https://www.anekdot.ru/rss/randomu.html"
_NAME_RE = re.compile(r"^[\w .\-]{2,24}$")

# Step 1 is one tap for the usual answer; step 2 (only after "no" for a found
# reference, "yes" for a missed candidate) asks what exactly, with buttons
# named by the values on screen.
STEP1 = {
    "verify": ("✓ Всё верно", "✗ Есть ошибка"),
    "missed": ("Да", "Нет"),
}
INFO = """
### ℹ️ Что это

Программа ищет в судебных решениях ссылки на статьи законов (`ч. 3 ст. 158 УК РФ`). Вы проверяете её работу.

- **Найденная ссылка.** Сравните выделенное с тем, как программа это поняла. Если что-то не так, нажмите «Есть ошибка» и выберите, что именно.
- **Пропущенная ссылка.** Скажите, есть ли в выделенном ссылка на статью, и если есть, то на закон или на другой документ (договор, правила, приказ).
- Сомневаетесь - «Пропустить». Среди вопросов есть проверочные с известным ответом.

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
.icon-btn { font-size: 1.35em !important; width: 44px; height: 44px; padding: 0 !important;
            border-radius: 50% !important; }
#title { flex: 1 1 auto; text-align: center; font-weight: 600; font-size: 1.05em; }
#item { font-size: 1.05em; line-height: 1.7; }
.answers { flex-wrap: wrap; gap: 8px; }
.answers button { flex: 1 1 140px; min-height: 48px; }
.choices { gap: 8px; }
.choices button { flex: 0 0 auto !important; min-height: 48px; height: auto; }
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
# Packs in file order; the newest pack is asked first, until every item of
# it has TARGET_VOTES votes (survey1 items have no "pack" field).
_PACKS = list(dict.fromkeys(it.get("pack", "survey1") for it in ITEMS.values()))
_PACK_RANK = {i: _PACKS.index(it.get("pack", "survey1")) for i, it in ITEMS.items()}
TARGET_VOTES = 2

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
        def priority(i: str) -> Tuple[bool, int, int]:
            votes = _votes_per_item[i]
            return votes >= TARGET_VOTES, -_PACK_RANK[i], votes

        best = min(priority(i) for i in left)
        return random.choice([i for i in left if priority(i) == best])


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
        return "<h3>📊 Рейтинг</h3><p>Пока никто не ответил.</p>"
    return (
        "<h3>📊 Рейтинг</h3><table style='width:100%'><tr><th>#</th><th>Имя</th><th>Ответов</th></tr>"
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
        question = "<p style='margin-top: 14px;'><b>Здесь есть ссылка на статью?</b></p>"
    else:
        question = (
            "<p style='margin-top: 14px; margin-bottom: 4px;'>Программа поняла выделенное так:</p>"
            '<div style="margin: 4px 0 10px; padding: 8px 10px; border-left: 3px solid rgb(250, 204, 21);">'
            f"<b>{html.escape(it['claim'])}</b></div><p><b>Всё верно?</b></p>"
        )
    return f"<div>{text}</div>{question}"


def step2_choices(item: Dict[str, object]):
    """Prompt, (answer, label) pairs of the second step."""
    if item["kind"] == "missed":
        return "Ссылка на что?", [
            ("yes", "Статью закона или кодекса"),
            ("other_doc", "Пункт другого документа: договора, правил, приказа"),
        ]
    f = item.get("fields") or {}
    return "Что неверно?", [
        ("no_law", f"Закон: {f.get('law') or 'не указан'}"),
        ("no_article", f"Статья: {f.get('article') or 'не указана'}"),
        ("no_part", f"Часть, пункт: {f.get('lower') or 'не указаны'}"),
        ("no_ref", "Это вообще не ссылка"),
    ]


def screen(state: Dict[str, object], step: int, joke: Optional[str] = None):
    """Values for every component of the question area (see OUTPUTS)."""
    item_id = state.get("item")
    item = ITEMS[item_id] if item_id else None
    state["step"] = step
    count = len(_answered[state["voter"]])
    yes_label, no_label = STEP1["missed" if item and item["kind"] == "missed" else "verify"]
    prompt, choices = step2_choices(item) if item else ("", [])
    state["choices"] = [v for v, _ in choices]
    choice_updates = [
        gr.update(value=choices[i][1], visible=True) if i < len(choices) else gr.update(visible=False)
        for i in range(4)
    ]
    return (
        state,
        card(item_id),
        f"{html.escape(state.get('name', ''))}, ваших ответов: {count}",
        gr.update(visible=bool(item) and step == 1),
        gr.update(visible=bool(item) and step == 2),
        gr.update(value=yes_label),
        gr.update(value=no_label),
        f"**{prompt}**",
        *choice_updates,
        gr.update(visible=True) if joke else gr.skip(),
        joke if joke else gr.skip(),
    )


def show(state: Dict[str, object]):
    """Move to the next item for the voter."""
    state["item"] = next_item(state["voter"])
    state["shown_at"] = time.time()
    return screen(state, 1)


def start(stored: Optional[Dict[str, object]]):
    voter = (stored or {}).get("voter") or uuid.uuid4().hex
    name = (stored or {}).get("name") or ""
    state = {"voter": voter, "name": name, "item": None, "shown_at": time.time()}
    if name and claim_name(voter, name)[0]:
        return (*show(state)[:N_SCREEN], gr.update(visible=False), "", {"voter": voter, "name": name})
    return (*screen(state, 1)[:N_SCREEN], gr.update(visible=True), "", {"voter": voter, "name": ""})


def set_name(state: Dict[str, object], name: str):
    ok, result = claim_name(state["voter"], name)
    if not ok:
        return (*[gr.skip()] * N_SCREEN, gr.update(visible=True),
                f"<p style='color: #d9534f;'>{html.escape(result)}</p>", gr.skip())
    state["name"] = result
    return (*show(state)[:N_SCREEN], gr.update(visible=False), "", {"voter": state["voter"], "name": result})


def joke_frame() -> str:
    """The anekdot.ru informer as its authors intend it (their script, their
    attribution link), isolated in an iframe; jokes are not stored here."""
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'><base target='_blank'><style>"
        "body{margin:0;padding:4px;font:16px/1.5 system-ui,sans-serif;color:#1f2328;background:transparent}"
        "a{color:#b45309}#a_rnd_title{font-size:13px}#a_rnd_next{display:none}"
        "@media (prefers-color-scheme:dark){body{color:#e6e6e6}a{color:#facc15}}"
        "#loading{opacity:.6}body:has(#a_rnd) #loading{display:none}"
        "</style></head><body><div id='loading'>Загружаем анекдот…</div>"
        f"<script src='{JOKE_INFORMER}?r={random.randint(1, 10**9)}'></script></body></html>"
    )
    return (
        f'<iframe srcdoc="{html.escape(doc, quote=True)}" title="Анекдот" '
        'style="width:100%;height:200px;border:0;background:transparent"></iframe>'
    )


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
    state["item"] = next_item(state["voter"])
    state["shown_at"] = time.time()
    count = len(_answered[state["voter"]])
    next_joke = state.setdefault("next_joke", count + random.randint(*JOKE_GAP))
    if item_id and count >= next_joke:
        state["next_joke"] = count + random.randint(*JOKE_GAP)
        return screen(state, 1, joke=joke_frame())
    return screen(state, 1)


def _kind(state: Dict[str, object]) -> Optional[str]:
    item_id = state.get("item")
    return ITEMS[item_id]["kind"] if item_id else None


def on_yes(state):
    return screen(state, 2) if _kind(state) == "missed" else answer(state, "yes")


def on_no(state):
    return answer(state, "no") if _kind(state) == "missed" else screen(state, 2)


def on_choice(state, i: int):
    choices = state.get("choices") or []
    return answer(state, choices[i]) if i < len(choices) else screen(state, 2)


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
            board_btn = gr.Button("📊", size="sm", scale=0, min_width=44, elem_classes=["icon-btn"])
            gr.HTML("<div>⚖️ Проверка юридических ссылок</div>", elem_id="title")
            info_btn = gr.Button("ℹ️", size="sm", scale=0, min_width=44, elem_classes=["icon-btn"])
        progress = gr.Markdown()
        item_html = gr.HTML(elem_id="item")
        with gr.Row(visible=False, elem_classes=["answers"]) as step1_row:
            yes_btn = gr.Button("✓ Всё верно", variant="primary")
            no_btn = gr.Button("✗ Есть ошибка")
            skip_btn = gr.Button("Пропустить")
        with gr.Column(visible=False) as step2_col:
            prompt = gr.Markdown()
            with gr.Column(elem_classes=["choices"]):
                choice_btns = [gr.Button("", visible=False) for _ in range(4)]
            back_btn = gr.Button("← Назад", size="sm")

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
        with gr.Column(visible=False, elem_classes=["overlay"]) as joke_modal:
            with gr.Column(elem_classes=["overlay-card"]):
                gr.Markdown("### 😄 Перерыв на анекдот\nСпасибо, что отвечаете! Случайный анекдот с [anekdot.ru](https://www.anekdot.ru/).")
                joke_html = gr.HTML()
                joke_close = gr.Button("Дальше", variant="primary")
        with gr.Column(visible=False, elem_classes=["overlay"]) as board_modal:
            with gr.Column(elem_classes=["overlay-card"]):
                board_html = gr.HTML()
                board_close = gr.Button("Закрыть", variant="primary")

        outputs = [state, item_html, progress, step1_row, step2_col, yes_btn, no_btn, prompt,
                   *choice_btns, joke_modal, joke_html]
        yes_btn.click(on_yes, [state], outputs)
        no_btn.click(on_no, [state], outputs)
        skip_btn.click(lambda s: answer(s, "unsure"), [state], outputs)
        back_btn.click(lambda s: screen(s, 1), [state], outputs)
        for i, btn in enumerate(choice_btns):
            btn.click(lambda s, i=i: on_choice(s, i), [state], outputs)
        screen_outputs = outputs[:N_SCREEN]
        for trigger in (name_btn.click, name_box.submit):
            trigger(set_name, [state, name_box], [*screen_outputs, name_modal, name_error, stored])
        info_btn.click(lambda: gr.update(visible=True), None, info_modal)
        info_close.click(lambda: gr.update(visible=False), None, info_modal)
        board_btn.click(lambda s: (leaderboard(s.get("voter")), gr.update(visible=True)), [state],
                        [board_html, board_modal])
        board_close.click(lambda: gr.update(visible=False), None, board_modal)
        joke_close.click(lambda: (gr.update(visible=False), ""), None, [joke_modal, joke_html])
        app.load(start, [stored], [*screen_outputs, name_modal, name_error, stored])
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

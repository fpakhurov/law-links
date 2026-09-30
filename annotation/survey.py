"""Yes/no survey items: is a found reference right, is a candidate a missed one.

Items are built offline with the frozen extractor, the survey app (space/)
only shows them and stores answers.

    verify   a reference the extractor found: the fragment in context and
             its reading ("Статья 158, часть 3 - Уголовный кодекс РФ").
             Answers: yes | no_law | no_numbers | no_ref | unsure.
    missed   a candidate the extractor did not return: a chain without a
             resolved law, or a marker with a number outside any chain.
             Answers: yes | other_doc | no | unsure ("other_doc": a point of
             a contract, rules, an order - not a missed law reference).
             Gives recall relative to the pool.
    control  a verify item with a known answer, from the dev gold sets:
             a correct reading (expected yes) or a deliberately corrupted
             one (another article number or another code, expected no).
             Used to screen out careless voters.

Usage:
    python -m annotation.survey build survey1 --docs 40 --controls 40
    python -m annotation.survey space survey1      # copy items into space/
    python -m annotation.survey analyze survey1 --votes path/to/votes

Analysis: voters are kept if they answered at least MIN_CONTROLS control
items with accuracy >= MIN_CONTROL_ACCURACY ("unsure" not counted). Every
item gets the majority answer of kept voters, "unsure" ignored, a tie is
undecided. Precision = verify items judged right / decided verify items,
with a Wilson interval; recall is relative to the candidate pool: right
verify items / (right verify items + missed candidates judged "yes").
"""

import argparse
import itertools
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from annotation.laws import load_titles
from annotation.make_tasks import used_doc_ids
from annotation.store import DATA_DIR
from law_links import DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH, ROOT
from law_links.extractor import DetectedLink, Extractor
from law_links.normalize import normalize
from research.corpus import CORPUS_DIR, INDEX_PATH
from research.sample_test import window
from scripts.eval import load_cases, to_key

SURVEY_DIR = DATA_DIR / "survey"
CONTEXT_CHARS = 150
_MARKER_RE = re.compile(
    r"(?<![а-яa-z])(?:ст|стать[а-я]*|пп?|пункт[а-я]*|подп|подпункт[а-я]*|ч|част[а-я]*)\.?\s*\d[\d.]*",
    re.IGNORECASE,
)
# The resolver keeps the shortest of equally good name prefixes ("ст. 15 УПК"
# for "ст. 15 УПК РФ"); for display the fragment takes the country suffix too.
_LAW_TAIL_RE = re.compile(r"\s+(?:РФ|России|Российской\s+Федерации)(?![А-Яа-яЁё])")
# Codes used to corrupt control items: the reading names another code.
_CODE_SWAP = {5: 13, 13: 5, 15: 10, 10: 15, 17: 12, 12: 17, 0: 6, 6: 0, 7: 8, 8: 7}


def context(text: str, start: int, end: int, chars: int = CONTEXT_CHARS) -> Tuple[str, str, str]:
    """Text before, the fragment, text after; cut at whitespace."""
    left = max(0, start - chars)
    right = min(len(text), end + chars)
    while 0 < left < start and not text[left].isspace() and not text[left - 1].isspace():
        left -= 1  # a cut word is shown whole
    while end < right < len(text) and not text[right - 1].isspace() and not text[right].isspace():
        right += 1
    before = ("… " + text[left:start].lstrip()) if left > 0 else text[:start]
    after = (text[end:right].rstrip() + " …") if right < len(text) else text[end:]
    return before, text[start:end], after


def with_law_tail(text: str, end: int) -> int:
    match = _LAW_TAIL_RE.match(text, end)
    return match.end() if match else end


def _values(values: Sequence[Optional[str]]) -> List[str]:
    return list(dict.fromkeys(v for v in values if v))


def claim(links: Sequence[Dict[str, object]], titles: Dict[int, str]) -> str:
    """Reading of one chain with numbers in the order of the citation and no
    level names, so it can be compared with the text without knowing the
    labeling rules: "п. 6 ч. 1 ст. 24.5 КоАП РФ" -> "6 · 1 · ст. 24.5 — КоАП"."""
    arts = _values([l["article"] for l in links])
    pts = _values([l["point_article"] for l in links])
    subs = _values([l["subpoint_article"] for l in links])
    parts = [", ".join(v) for v in (subs, pts) if v]
    parts.append("ст. " + ", ".join(arts) if arts else "статья не указана")
    law = int(links[0]["law_id"])
    return " · ".join(parts) + " — " + titles.get(law, f"закон {law}")

def group_detected(detected: Sequence[DetectedLink]) -> List[Tuple[int, int, List[Dict[str, object]]]]:
    """One group per chain: (start, end, links)."""
    groups: Dict[Tuple[int, int, int], List[Dict[str, object]]] = {}
    for d in detected:
        groups.setdefault((d.start, d.end, d.link.law_id), []).append(d.link.model_dump())
    return [(s, e, links) for (s, e, _), links in groups.items()]


def doc_items(extractor: Extractor, doc_id: str, text: str, source: str, titles) -> List[Dict[str, object]]:
    norm = normalize(text)
    items = []
    covered: List[Tuple[int, int]] = []
    for n, (start, end, links) in enumerate(group_detected(extractor.extract_detailed(text))):
        end = with_law_tail(text, end)
        covered.append((start, end))
        before, fragment, after = context(text, start, end)
        items.append({
            "item_id": f"v-{doc_id}-{n}", "kind": "verify", "doc_id": doc_id, "source": source,
            "before": before, "fragment": fragment, "after": after,
            "claim": claim(links, titles), "links": links, "expected": None,
        })

    def outside(s: int, e: int) -> bool:
        return all(e <= cs or s >= ce for cs, ce in covered)

    # a candidate needs a number: letter-only "chains" are mostly initials ("П.Ю.")
    candidates = [
        (c.start, c.end) for c in extractor.chains.find(norm)
        if outside(c.start, c.end) and any(ch.isdigit() for ch in norm[c.start : c.end])
    ]
    for s, e in candidates:
        covered.append((s, e))
    candidates += [(m.start(), m.end()) for m in _MARKER_RE.finditer(norm) if outside(m.start(), m.end())]
    for n, (start, end) in enumerate(sorted(candidates)):
        before, fragment, after = context(text, start, end)
        items.append({
            "item_id": f"m-{doc_id}-{n}", "kind": "missed", "doc_id": doc_id, "source": source,
            "before": before, "fragment": fragment, "after": after,
            "claim": None, "links": [], "expected": None,
        })
    return items


def control_items(extractor: Extractor, titles, n: int, rng: random.Random) -> List[Dict[str, object]]:
    """Correct readings from the dev gold sets and corrupted copies of them."""
    correct = []
    for path in (ROOT / "tests" / "gold.json", ROOT / "tests" / "gold_real.json"):
        for case in load_cases(path):
            gold = Counter(to_key(l) for l in case["links"])
            for start, end, links in group_detected(extractor.extract_detailed(case["text"])):
                pred = Counter(to_key(l) for l in links)
                if not pred - gold and links[0]["article"]:
                    correct.append((case, start, end, links))
    rng.shuffle(correct)
    items = []
    for i, (case, start, end, links) in enumerate(correct[:n]):
        before, fragment, after = context(case["text"], start, with_law_tail(case["text"], end))
        corrupt = i % 2 == 1
        shown = [dict(l) for l in links]
        if corrupt:
            law = int(shown[0]["law_id"])
            if law in _CODE_SWAP and i % 4 == 1:
                for l in shown:
                    l["law_id"] = _CODE_SWAP[law]
            else:
                bump = rng.choice([1, 2, 3, 7, 10])
                for l in shown:
                    head, _, tail = str(l["article"]).partition(".")
                    if head.isdigit():
                        l["article"] = str(int(head) + bump) + ("." + tail if tail else "")
        items.append({
            "item_id": f"c-{case['id']}-{start}", "kind": "control", "doc_id": case["id"],
            "source": case.get("source", "tests/" + case["id"]),
            "before": before, "fragment": fragment, "after": after,
            "claim": claim(shown, titles), "links": shown,
            "expected": "no" if corrupt else "yes",
        })
    return items


def build(name: str, docs: int, controls: int, window_chars: int, seed: int) -> Path:
    rng = random.Random(seed)
    titles = load_titles()
    extractor = Extractor.from_files(DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH)
    used = used_doc_ids()
    records = [json.loads(l) for l in INDEX_PATH.read_text("utf-8").splitlines()]
    pool = sorted((r for r in records if r["id"] not in used), key=lambda r: r["id"])
    picked = rng.sample(pool, min(docs, len(pool)))
    items: List[Dict[str, object]] = []
    for r in picked:
        lines = (CORPUS_DIR / f"{r['id']}.txt").read_text("utf-8").splitlines()
        start, stop = window(lines, rng, window_chars)
        text = "\n".join(lines[start:stop])
        source = f"{r['url']} (строки {start + 1}-{stop})"
        items += doc_items(extractor, r["id"], text, source, titles)
    items += control_items(extractor, titles, controls, rng)
    out_dir = SURVEY_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "items.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    (out_dir / "docs.json").write_text(json.dumps([r["id"] for r in picked], indent=1) + "\n", "utf-8")
    kinds = Counter(i["kind"] for i in items)
    print(f"{len(items)} items from {len(picked)} documents: {dict(kinds)} -> {path}")
    return path


MIN_CONTROLS = 3
MIN_CONTROL_ACCURACY = 0.75
_OUT_OF_DICT_RE = re.compile(r"Конституци", re.IGNORECASE)
NO_ANSWERS = {"no", "no_law", "no_numbers", "no_ref", "other_doc"}


def load_votes(folder: Path) -> List[Dict[str, object]]:
    votes, seen = [], set()
    for path in sorted(Path(folder).rglob("votes-*.jsonl")):
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                vote = json.loads(line)
                if vote["vote_id"] not in seen:
                    seen.add(vote["vote_id"])
                    votes.append(vote)
    return votes


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_correct(answer: str, expected: str) -> bool:
    return answer == "yes" if expected == "yes" else answer in NO_ANSWERS


def screen_voters(items: Dict[str, Dict], votes: List[Dict]) -> Dict[str, Dict[str, object]]:
    """Control accuracy per voter and whether the voter is kept."""
    stats: Dict[str, Dict[str, object]] = {}
    for v in votes:
        s = stats.setdefault(v["voter"], {"answers": 0, "controls": 0, "right": 0, "ms": []})
        s["answers"] += 1
        s["ms"].append(v.get("ms", 0))
        item = items.get(v["item_id"])
        if item and item["kind"] == "control" and v["answer"] != "unsure":
            s["controls"] += 1
            s["right"] += is_correct(v["answer"], item["expected"])
    for s in stats.values():
        s["accuracy"] = s["right"] / s["controls"] if s["controls"] else None
        s["kept"] = s["controls"] >= MIN_CONTROLS and s["accuracy"] >= MIN_CONTROL_ACCURACY
        ms = sorted(s["ms"])
        s["median_ms"] = ms[len(ms) // 2] if ms else 0
    return stats


def decide(answers: List[str]) -> Optional[str]:
    """Majority answer: "yes", "no" (with the most common reason) or None."""
    answers = [a for a in answers if a != "unsure"]
    yes = sum(a == "yes" for a in answers)
    no = [a for a in answers if a in NO_ANSWERS]
    if yes > len(no):
        return "yes"
    if len(no) > yes:
        return Counter(no).most_common(1)[0][0]
    return None


def analyze(items: Dict[str, Dict], votes: List[Dict]) -> Dict[str, object]:
    voters = screen_voters(items, votes)
    kept = {v for v, s in voters.items() if s["kept"]}
    by_item: Dict[str, List[str]] = {}
    for v in votes:
        if v["voter"] in kept and items.get(v["item_id"], {}).get("kind") in ("verify", "missed"):
            by_item.setdefault(v["item_id"], []).append(v["answer"])
    decisions = {i: decide(a) for i, a in by_item.items()}

    verify = {i: d for i, d in decisions.items() if items[i]["kind"] == "verify" and d}
    right = [i for i, d in verify.items() if d == "yes"]
    errors = Counter(d for d in verify.values() if d != "yes")
    links_right = sum(len(items[i]["links"]) for i in right)
    links_all = sum(len(items[i]["links"]) for i in verify)
    missed = {i: d for i, d in decisions.items() if items[i]["kind"] == "missed" and d}
    # A reference to an act missing from law_aliases.json (the Constitution
    # first of all) is not a miss of the extractor under the task rules.
    outside = [i for i, d in missed.items() if d == "yes" and _OUT_OF_DICT_RE.search(items[i]["after"][:60])]
    missed_yes = [i for i, d in missed.items() if d == "yes" and i not in outside]

    pairs = agree = 0
    for answers in by_item.values():
        binary = ["yes" if a == "yes" else "no" for a in answers if a != "unsure"]
        for a, b in itertools.combinations(binary, 2):
            pairs += 1
            agree += a == b
    return {
        "voters": voters,
        "kept": len(kept),
        "verify_decided": len(verify),
        "precision": len(right) / len(verify) if verify else None,
        "precision_ci": wilson(len(right), len(verify)),
        "link_precision": links_right / links_all if links_all else None,
        "errors": dict(errors),
        "missed_decided": len(missed),
        "missed_yes": len(missed_yes),
        "missed_out_of_dictionary": len(outside),
        "relative_recall": len(right) / (len(right) + len(missed_yes)) if right or missed_yes else None,
        "agreement": agree / pairs if pairs else None,
        "pairs": pairs,
        "decisions": decisions,
    }


def report(name: str, result: Dict[str, object]) -> str:
    fmt = lambda x: "-" if x is None else f"{x:.3f}"
    lo, hi = result["precision_ci"]
    lines = [
        f"# Опрос {name}: результаты", "",
        f"- участников: {len(result['voters'])}, прошли контроль: {result['kept']}",
        f"- согласие участников (попарно, да/нет): {fmt(result['agreement'])} на {result['pairs']} парах",
        f"- точность по цепочкам: {fmt(result['precision'])} (95% ДИ {lo:.3f}-{hi:.3f}) на {result['verify_decided']} решённых вопросах",
        f"- точность по ссылкам (вес - число ссылок в цепочке): {fmt(result['link_precision'])}",
        f"- ошибки: {result['errors']}",
        f"- кандидаты на пропуск: {result['missed_yes']} «да» из {result['missed_decided']} решённых"
        f" (ещё {result['missed_out_of_dictionary']} - акты вне словаря, например Конституция, не считаются)",
        f"- полнота относительно пула кандидатов: {fmt(result['relative_recall'])}",
        "", "| участник | ответов | контрольных | точность контроля | медиана, с | учтён |", "|---|---|---|---|---|---|",
    ]
    for voter, s in sorted(result["voters"].items(), key=lambda x: -x[1]["answers"]):
        lines.append(
            f"| {voter[:8]} | {s['answers']} | {s['controls']} | {fmt(s['accuracy'])} | "
            f"{s['median_ms'] / 1000:.1f} | {'да' if s['kept'] else 'нет'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--docs", type=int, default=40)
    b.add_argument("--controls", type=int, default=40)
    b.add_argument("--window", type=int, default=6000)
    b.add_argument("--seed", type=int, default=20261002)
    sp = sub.add_parser("space")
    sp.add_argument("name")
    an = sub.add_parser("analyze")
    an.add_argument("name")
    an.add_argument("--votes", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        build(args.name, args.docs, args.controls, args.window, args.seed)
    elif args.cmd == "space":
        target = ROOT / "space" / "items.jsonl"
        shutil.copyfile(SURVEY_DIR / args.name / "items.jsonl", target)
        print(f"copied to {target}")
    elif args.cmd == "analyze":
        survey_dir = SURVEY_DIR / args.name
        items = {
            i["item_id"]: i
            for i in map(json.loads, (survey_dir / "items.jsonl").read_text("utf-8").splitlines())
        }
        result = analyze(items, load_votes(args.votes))
        text = report(args.name, result)
        (survey_dir / "report.md").write_text(text, "utf-8")
        with open(survey_dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for item_id, decision in sorted(result["decisions"].items()):
                if decision:
                    f.write(json.dumps({"item_id": item_id, "label": decision}, ensure_ascii=False) + "\n")
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

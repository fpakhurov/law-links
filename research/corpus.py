"""Unlabeled corpus of court decisions from sudact.ru.

Crawls practice categories -> topics -> topic pages -> decisions and stores
plain text, one file per decision. Used to train statistical models
(language models, embeddings) and to sample the held-out test set.
Documents already used in gold sets are skipped.

Usage:
    python -m research.corpus [--pages 1] [--limit 600] [--delay 1.0]
    python -m research.corpus --docs arbitral/KU05IYO5Xxm9 ... [--topic search]

Practice topic pages link only to courts of general jurisdiction, so
arbitration decisions are added by explicit ids (found by web search).
"""

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional

from law_links import ROOT

BASE = "https://sudact.ru"
CATEGORIES = [
    "sudebnaya-praktika-po-ugolovnym-delam",
    "sudebnaya-praktika-po-grazhdanskim-delam",
    "sudebnaya-praktika-po-administrativnym-delam",
]
# Topics not reachable from the category pages but known to exist.
EXTRA_TOPICS = ["po-dogovoru-postavki", "po-vosstanovleniyu-na-rabote"]
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) law-links-research"
DATA_DIR = ROOT / "research" / "data"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_PATH = DATA_DIR / "corpus_index.jsonl"

_TOPIC_RE = re.compile(r'href="/practice/([a-z0-9-]+)/"')
_DOC_RE = re.compile(r'href="/(regular|arbitral|vsrf|magistrate)/doc/([A-Za-z0-9]+)/')
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
_FOOTER_RE = re.compile(r"^\s*Суд:.*\(подробнее\)", re.M)


def fetch(url: str, delay: float) -> Optional[str]:
    time.sleep(delay)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # network errors are logged and skipped
        print(f"skip {url}: {exc}", file=sys.stderr)
        return None


def decision_text(page: str) -> str:
    """Plain text of a decision page: from the heading rule to the end of
    the document block, line breaks preserved."""
    start = page.find('class="hr-h1"')
    if start < 0:
        return ""
    body = page[page.find(">", start) + 1 :]
    end = body.find('<div class="b-share')
    if end > 0:
        body = body[:end]
    body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    body = re.sub(r"<br\s*/?>", "\n", body)
    body = re.sub(r"<[^>]+>", "", body)
    text = html.unescape(body)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return strip_footer(text).strip()


def strip_footer(text: str) -> str:
    """Drop the site footer ("Суд: ...", "Судебная практика по: ...")."""
    match = _FOOTER_RE.search(text)
    return text[: match.start()] if match else text


def test_doc_ids() -> set:
    manifest = ROOT / "tests" / "test_texts" / "manifest.json"
    if not manifest.exists():
        return set()
    return {r["id"] for r in json.loads(manifest.read_text("utf-8"))}


def training_texts(limit: Optional[int] = None) -> List[str]:
    """Corpus texts for unsupervised training; held-out test documents are
    excluded so no model sees them before the final evaluation."""
    held_out = test_doc_ids()
    paths = [p for p in sorted(CORPUS_DIR.glob("*.txt")) if p.stem not in held_out]
    return [p.read_text("utf-8") for p in paths[:limit]]


def gold_doc_ids() -> set:
    ids = set()
    for path in (ROOT / "tests").glob("gold*.json"):
        for case in json.loads(path.read_text("utf-8"))["cases"]:
            match = re.search(r"/doc/([A-Za-z0-9]+)/", case.get("source", ""))
            if match:
                ids.add(match.group(1))
    return ids


def topics(delay: float) -> List[str]:
    found: List[str] = []
    for category in CATEGORIES:
        page = fetch(f"{BASE}/practice/{category}/", delay) or ""
        for slug in _TOPIC_RE.findall(page):
            if slug not in found and slug not in CATEGORIES:
                found.append(slug)
    for slug in EXTRA_TOPICS:
        if slug not in found:
            found.append(slug)
    return found


def doc_refs(topic_slugs: Iterable[str], pages: int, delay: float) -> List[tuple]:
    refs: List[tuple] = []
    seen = set()
    for slug in topic_slugs:
        for page_no in range(1, pages + 1):
            suffix = f"?page={page_no}" if page_no > 1 else ""
            page = fetch(f"{BASE}/practice/{slug}/{suffix}", delay) or ""
            for kind, doc_id in _DOC_RE.findall(page):
                if doc_id not in seen:
                    seen.add(doc_id)
                    refs.append((kind, doc_id, slug))
    return refs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--docs", nargs="*", default=None, help="kind/id pairs")
    parser.add_argument("--topic", default="search")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    excluded = gold_doc_ids()
    done = {p.stem for p in CORPUS_DIR.glob("*.txt")}
    if args.docs:
        refs = [(*d.split("/"), args.topic) for d in args.docs]
        refs = [r for r in refs if r[1] not in excluded]
    else:
        slugs = topics(args.delay)
        print(f"topics: {len(slugs)}")
        refs = [r for r in doc_refs(slugs, args.pages, args.delay) if r[1] not in excluded]
    print(f"documents listed: {len(refs)}, excluded gold: {len(excluded)}")

    with open(INDEX_PATH, "a", encoding="utf-8") as index:
        for kind, doc_id, slug in refs[: args.limit]:
            if doc_id in done:
                continue
            page = fetch(f"{BASE}/{kind}/doc/{doc_id}/", args.delay)
            if page is None:
                continue
            text = decision_text(page)
            if len(text) < 1000:
                continue
            (CORPUS_DIR / f"{doc_id}.txt").write_text(text + "\n", "utf-8")
            title = _TITLE_RE.search(page)
            record = {
                "id": doc_id,
                "kind": kind,
                "topic": slug,
                "url": f"{BASE}/{kind}/doc/{doc_id}/",
                "title": html.unescape(title.group(1)).strip() if title else "",
                "chars": len(text),
            }
            index.write(json.dumps(record, ensure_ascii=False) + "\n")
            index.flush()
            done.add(doc_id)
    print(f"corpus size: {len(done)} documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Create a batch of annotation tasks from the corpus.

Random documents (fixed seed) that nobody has seen yet: documents of the
gold sets, of earlier batches and the ones read during development are
excluded. From each document a random window of whole lines is taken, so
the batch follows the real distribution of text.

Usage:
    python -m annotation.make_tasks test2 --regular 9 --arbitral 3 --mode blind
"""

import argparse
import json
import random
import re
import sys

from annotation.store import TASKS_DIR, Task, batches, load_tasks, write_tasks
from research.corpus import CORPUS_DIR, INDEX_PATH, gold_doc_ids, test_doc_ids
from research.sample_test import SEEN, window


def used_doc_ids() -> set:
    used = set(SEEN) | gold_doc_ids() | test_doc_ids()
    for batch in batches():
        used.update(t.doc_id for t in load_tasks(batch))
    return used


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch")
    parser.add_argument("--regular", type=int, default=9)
    parser.add_argument("--arbitral", type=int, default=3)
    parser.add_argument("--mode", choices=["blind", "suggest"], default="blind")
    parser.add_argument("--window", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9_-]+", args.batch):
        parser.error("batch name: lowercase letters, digits, '-' and '_'")
    if (TASKS_DIR / f"{args.batch}.jsonl").exists():
        parser.error(f"batch {args.batch} already exists")

    rng = random.Random(args.seed)
    used = used_doc_ids()
    records = [json.loads(l) for l in INDEX_PATH.read_text("utf-8").splitlines()]
    records = [r for r in records if r["id"] not in used]
    picked = []
    for kind, n in (("regular", args.regular), ("arbitral", args.arbitral)):
        pool = sorted((r for r in records if r["kind"] == kind), key=lambda r: r["id"])
        picked += rng.sample(pool, min(n, len(pool)))

    tasks = []
    for r in picked:
        lines = (CORPUS_DIR / f"{r['id']}.txt").read_text("utf-8").splitlines()
        start, stop = window(lines, rng, args.window)
        tasks.append(Task(
            doc_id=r["id"],
            text="\n".join(lines[start:stop]) + "\n",
            source=f"{r['url']} (строки {start + 1}-{stop}, тема {r['topic']})",
            mode=args.mode,
        ))
    path = write_tasks(args.batch, tasks)
    print(f"{len(tasks)} tasks -> {path} (excluded {len(used)} used documents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sample documents for the held-out test set.

Random documents from the corpus (fixed seed), a random contiguous window
of whole lines from each, so the test follows the real distribution of
text rather than the densest reasoning parts. Documents seen during
development are excluded.

Usage:
    python -m research.sample_test [--seed 20260929] [--regular 6] [--arbitral 2]
"""

import argparse
import json
import random
import sys

from law_links import ROOT
from research.corpus import CORPUS_DIR, INDEX_PATH

# Downloaded and looked at while building gold_real, not annotated.
SEEN = {"fXT2G1xGTMLO", "bcf17nT5wdZl", "4E166mtgtbZO", "ugLY7CxApcxJ", "Ege3s6Uz0UVr"}
OUT_DIR = ROOT / "tests" / "test_texts"
WINDOW_CHARS = 8000


def window(lines, rng: random.Random, size: int):
    if sum(len(l) + 1 for l in lines) <= size:
        return 0, len(lines)
    ends = []
    total = 0
    for i, line in enumerate(lines):
        total += len(line) + 1
        ends.append(total)
    start = rng.randrange(len(lines))
    while start > 0 and ends[-1] - (ends[start - 1] if start else 0) < size:
        start -= 1
    base = ends[start - 1] if start else 0
    stop = start
    while stop < len(lines) and ends[stop] - base < size:
        stop += 1
    return start, min(stop + 1, len(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--regular", type=int, default=6)
    parser.add_argument("--arbitral", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = [json.loads(l) for l in INDEX_PATH.read_text("utf-8").splitlines()]
    records = [r for r in records if r["id"] not in SEEN]
    picked = []
    for kind, n in (("regular", args.regular), ("arbitral", args.arbitral)):
        pool = sorted((r for r in records if r["kind"] == kind), key=lambda r: r["id"])
        picked += rng.sample(pool, n)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for r in picked:
        lines = (CORPUS_DIR / f"{r['id']}.txt").read_text("utf-8").splitlines()
        start, stop = window(lines, rng, WINDOW_CHARS)
        name = f"{r['kind']}_{r['id']}.txt"
        (OUT_DIR / name).write_text("\n".join(lines[start:stop]) + "\n", "utf-8")
        manifest.append({**r, "file": name, "lines": [start + 1, stop]})
        print(f"{name}: lines {start + 1}-{stop} of {len(lines)}, topic {r['topic']}")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", "utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

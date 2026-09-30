"""Speed check against the benchmark limit: 4 benchmarks x 500 texts in 5 minutes.

Builds N texts from sentences of the real gold texts and the task text
(1 to 40 sentences each) and sends them to /detect one by one, as the
graders do. Also reports the worst case: a long text with every law name
seen for the first time (empty resolver cache).

Usage:
    python -m scripts.speed [--n 2000] [--seed 1]
"""

import argparse
import random
import re
import sys
import time

from fastapi.testclient import TestClient

from law_links import ROOT
from main import app
from scripts.eval import load_cases

LIMIT_SECONDS = 300


def build_texts(n: int, seed: int):
    texts = [c["text"] for c in load_cases(ROOT / "tests" / "gold_real.json") + load_cases(ROOT / "tests" / "gold_test.json")]
    texts.append((ROOT / "tests" / "readme_text.txt").read_text("utf-8"))
    sentences = [s for t in texts for s in re.split(r"(?<=[.!?])\s+", t) if 20 < len(s) < 1500]
    rng = random.Random(seed)
    return [" ".join(rng.sample(sentences, rng.choice([1, 3, 8, 20, 40]))) for _ in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    texts = build_texts(args.n, args.seed)
    readme = (ROOT / "tests" / "readme_text.txt").read_text("utf-8")
    with TestClient(app) as client:
        extractor = app.state.extractor
        extractor.resolver._best.clear()
        started = time.perf_counter()
        client.post("/detect", json={"text": readme})
        cold = time.perf_counter() - started

        started = time.perf_counter()
        for text in texts:
            client.post("/detect", json={"text": text})
        total = time.perf_counter() - started
    avg_chars = sum(map(len, texts)) // len(texts)
    print(f"{len(texts)} texts, {avg_chars} chars on average: {total:.1f} s ({1000 * total / len(texts):.1f} ms/text)")
    print(f"worst case, task text with an empty cache: {1000 * cold:.0f} ms -> {cold * args.n:.0f} s for {args.n} such texts")
    print(f"limit: {LIMIT_SECONDS} s for 2000 texts")
    return 0


if __name__ == "__main__":
    sys.exit(main())

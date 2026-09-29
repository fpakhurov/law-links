"""Benchmark the extractor on the gold sets, with ablations.

Usage:
    python -m research.bench [variant ...] [--split dev|test] [--log]

Variants: "final" and ablations "final?anaphora=False", "final?nb=False",
"final?threshold=0.8". Methods compared during the research (regex grammar,
fuzzy matching, n-gram LMs, word2vec, CRF) are in commit 372b392.

Splits:
    dev   tests/gold.json + tests/gold_real.json, used for all tuning
    test  tests/gold_test.json, held out; run only for final candidates

With --log every result is appended to research/results.jsonl.
"""

import argparse
import ast
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from law_links import DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH, ROOT
from law_links.extractor import Extractor
from scripts.eval import evaluate, load_cases

SPLITS = {
    "dev": [ROOT / "tests" / "gold.json", ROOT / "tests" / "gold_real.json"],
    "test": [ROOT / "tests" / "gold_test.json"],
}
RESULTS_PATH = ROOT / "research" / "results.jsonl"

_base: List[Extractor] = []


def build(anaphora: bool = True, nb: bool = True, threshold: float = 0.85) -> Extractor:
    if not _base:
        _base.append(Extractor.from_files(DEFAULT_ALIASES_PATH, DEFAULT_CHAIN_MODEL_PATH))
    base = _base[0]
    base.resolver.threshold = threshold
    return Extractor(base.chains, base.resolver, base.anaphora if anaphora else None, base.context_nb if nb else None)


VARIANTS = {"final": build}


def parse_params(spec: str) -> Tuple[str, Dict[str, object]]:
    name, _, query = spec.partition("?")
    params: Dict[str, object] = {}
    for pair in filter(None, query.split("&")):
        key, _, value = pair.partition("=")
        params[key] = ast.literal_eval(value)
    return name, params


def load_split(split: str) -> List[Dict]:
    cases: List[Dict] = []
    for path in SPLITS[split]:
        cases.extend(load_cases(path))
    return cases


def run(spec: str, cases: List[Dict]) -> Dict:
    name, params = parse_params(spec)
    extractor = VARIANTS[name](**params)
    chars = sum(len(c["text"]) for c in cases)
    started = time.perf_counter()
    metrics = evaluate(extractor, cases)
    metrics["ms_per_1k_chars"] = 1000 * (time.perf_counter() - started) / (chars / 1000)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--split", choices=sorted(SPLITS), default="dev")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    cases = load_split(args.split)
    print(f"{'variant':<44} {'P':>6} {'R':>6} {'F1':>6} {'tp':>5} {'fp':>5} {'fn':>5} {'ms/1k':>7}")
    for spec in args.names or ["final"]:
        m = run(spec, cases)
        print(
            f"{spec:<44} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
            f"{m['tp']:5d} {m['fp']:5d} {m['fn']:5d} {m['ms_per_1k_chars']:7.2f}"
        )
        if args.log:
            record = {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "variant": spec,
                "split": args.split,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            }
            with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

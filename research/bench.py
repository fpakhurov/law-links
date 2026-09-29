"""Benchmark extractor variants on the gold sets.

Usage:
    python -m research.bench [names ...] [--split dev|test] [--log]

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
from typing import Callable, Dict, List, Tuple

from law_links import DEFAULT_ALIASES_PATH, ROOT
from law_links.aliases import AliasIndex
from research.pipeline import (
    ExactMentions,
    NearestRightLinker,
    Pipeline,
    RegexChains,
    TrieMentions,
)
from scripts.eval import evaluate, load_cases

SPLITS = {
    "dev": [ROOT / "tests" / "gold.json", ROOT / "tests" / "gold_real.json"],
    "test": [ROOT / "tests" / "gold_test.json"],
}
RESULTS_PATH = ROOT / "research" / "results.jsonl"

_cache: Dict[str, object] = {}


def alias_index() -> AliasIndex:
    if "index" not in _cache:
        _cache["index"] = AliasIndex.from_json(DEFAULT_ALIASES_PATH)
    return _cache["index"]


def tfidf_linker(**kw):
    from research.methods.tfidf import TfidfLinker

    return TfidfLinker(DEFAULT_ALIASES_PATH, **kw)


def fuzzy_linker(**kw):
    from research.methods.fuzzy import FuzzyLinker

    return FuzzyLinker(DEFAULT_ALIASES_PATH, **kw)


# Each variant takes keyword parameters: "s1_tfidf?threshold=0.5&max_tokens=10".
VARIANTS: Dict[str, Callable[..., object]] = {
    "b0_exact": lambda: Pipeline(
        RegexChains(), ExactMentions.from_json(DEFAULT_ALIASES_PATH), NearestRightLinker()
    ),
    "b1_rules": lambda: Pipeline(RegexChains(), TrieMentions(alias_index()), NearestRightLinker()),
    "s1_tfidf": lambda **kw: Pipeline(RegexChains(), NoMentions(), tfidf_linker(**kw)),
    "s2_fuzzy": lambda **kw: Pipeline(RegexChains(), NoMentions(), fuzzy_linker(**kw)),
}


class NoMentions:
    def find(self, text):
        return []


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
    elapsed = time.perf_counter() - started
    metrics["ms_per_1k_chars"] = 1000 * elapsed / (chars / 1000)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--split", choices=sorted(SPLITS), default="dev")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    names = args.names or list(VARIANTS)
    cases = load_split(args.split)
    print(f"{'variant':<44} {'P':>6} {'R':>6} {'F1':>6} {'tp':>5} {'fp':>5} {'fn':>5} {'ms/1k':>7}")
    for name in names:
        m = run(name, cases)
        print(
            f"{name:<44} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} "
            f"{m['tp']:5d} {m['fp']:5d} {m['fn']:5d} {m['ms_per_1k_chars']:7.2f}"
        )
        if args.log:
            record = {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "variant": name,
                "split": args.split,
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            }
            with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Evaluate an extractor against tests/gold.json.

Usage:
    python -m scripts.eval [--gold tests/gold.json ...] [--verbose]

By default both gold sets are evaluated: tests/gold.json (task text and
constructed cases) and tests/gold_real.json (court decisions, a contract).

Metrics are computed over link multisets (order-insensitive):
exact match of all four fields.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from law_links import DEFAULT_ALIASES_PATH, ROOT
from law_links.aliases import AliasIndex
from law_links.extractor import RuleBasedExtractor

Key = Tuple[object, object, object, object]
DEFAULT_GOLD = [ROOT / "tests" / "gold.json", ROOT / "tests" / "gold_real.json"]


def to_key(link: Dict) -> Key:
    return (
        link.get("law_id"),
        link.get("article"),
        link.get("point_article"),
        link.get("subpoint_article"),
    )


def load_cases(gold_path: Path) -> List[Dict]:
    with open(gold_path, "r", encoding="utf-8") as file:
        cases = json.load(file)["cases"]
    for case in cases:
        if "text_file" in case:
            case["text"] = (gold_path.parent / case["text_file"]).read_text("utf-8")
    return cases


def score(gold: Counter, pred: Counter) -> Tuple[int, int, int]:
    tp = sum((gold & pred).values())
    return tp, sum(pred.values()) - tp, sum(gold.values()) - tp


def evaluate(extractor, cases: List[Dict], verbose: bool = False) -> Dict[str, float]:
    tp = fp = fn = 0
    for case in cases:
        gold = Counter(to_key(link) for link in case["links"])
        pred = Counter(to_key(l.model_dump()) for l in extractor.extract(case["text"]))
        c_tp, c_fp, c_fn = score(gold, pred)
        tp, fp, fn = tp + c_tp, fp + c_fp, fn + c_fn
        if verbose and (c_fp or c_fn):
            print(f"[{case['id']}]")
            for key in (pred - gold).elements():
                print(f"  FP {key}")
            for key in (gold - pred).elements():
                print(f"  FN {key}")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, action="append", default=None)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gold_paths = args.gold or DEFAULT_GOLD
    extractor = RuleBasedExtractor(AliasIndex.from_json(args.aliases))
    for gold_path in gold_paths:
        metrics = evaluate(extractor, load_cases(gold_path), verbose=args.verbose)
        print(
            "{name}: precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
            "tp={tp} fp={fp} fn={fn}".format(name=gold_path.name, **metrics)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

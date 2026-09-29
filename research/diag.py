"""Print false positives and false negatives of a variant on a split.

Usage:
    python -m research.diag final [--split dev]
"""

import argparse
import sys
from collections import Counter

from research.bench import VARIANTS, load_split, parse_params
from scripts.eval import to_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant")
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    name, params = parse_params(args.variant)
    extractor = VARIANTS[name](**params)
    for case in load_split(args.split):
        gold = Counter(to_key(l) for l in case["links"])
        pred = Counter(to_key(l.model_dump()) for l in extractor.extract(case["text"]))
        fp, fn = pred - gold, gold - pred
        if fp or fn:
            print(f"=== {case['id']}")
            for k in sorted(fp.elements(), key=str):
                print("  FP", k)
            for k in sorted(fn.elements(), key=str):
                print("  FN", k)
    return 0


if __name__ == "__main__":
    sys.exit(main())

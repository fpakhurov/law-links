"""Grid search of variant parameters on dev.

Usage:
    python -m research.sweep s1_tfidf threshold=0.5,0.6,0.7 ngram_range=(2,4),(3,5)

Prints one line per combination, best F1 last.
"""

import ast
import itertools
import sys

from research.bench import load_split, run


def parse_values(spec: str):
    key, _, values = spec.partition("=")
    return key, list(ast.literal_eval(f"[{values}]"))


def main() -> int:
    variant, *grid = sys.argv[1:]
    keys, value_lists = zip(*(parse_values(g) for g in grid)) if grid else ((), ())
    cases = load_split("dev")
    results = []
    for values in itertools.product(*value_lists):
        spec = variant + ("?" + "&".join(f"{k}={v!r}" for k, v in zip(keys, values)) if keys else "")
        m = run(spec, cases)
        results.append((m["f1"], spec, m))
        print(
            f"{spec:<70} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"fp={m['fp']} fn={m['fn']}",
            flush=True,
        )
    best = max(results, key=lambda r: r[0])
    print(f"BEST {best[1]} F1={best[0]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

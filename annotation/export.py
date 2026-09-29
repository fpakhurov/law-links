"""Agreement between annotators and export of a batch to a gold file.

Agreement is F1 between the link multisets of two annotators over the
documents both of them labeled (symmetric, 1.0 = identical links).

Export rules for every document:
- a label by the annotator named `adjudicator` wins;
- otherwise, if all annotators of the document gave identical links, they
  are taken;
- otherwise the document is a conflict: listed, not exported.

Usage:
    python -m annotation.export test2 [--out tests/gold_test2.json]
"""

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from annotation.store import Label, annotators, label_links, load_labels, load_tasks

ADJUDICATOR = "adjudicator"


def key_counter(label: Label) -> Counter:
    return Counter(
        (l["law_id"], l["article"], l["point_article"], l["subpoint_article"]) for l in label_links(label)
    )


def pair_f1(a: Dict[str, Label], b: Dict[str, Label]) -> Tuple[float, int]:
    """F1 of b against a over shared documents, and the number of them."""
    shared = sorted(set(a) & set(b))
    tp = fp = fn = 0
    for doc in shared:
        ca, cb = key_counter(a[doc]), key_counter(b[doc])
        common = sum((ca & cb).values())
        tp, fp, fn = tp + common, fp + sum(cb.values()) - common, fn + sum(ca.values()) - common
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return f1, len(shared)


def agreement(batch: str) -> List[Tuple[str, str, float, int]]:
    people = [p for p in annotators(batch) if p != ADJUDICATOR]
    labels = {p: load_labels(batch, p) for p in people}
    return [(a, b, *pair_f1(labels[a], labels[b])) for a, b in itertools.combinations(people, 2)]


def export(batch: str) -> Tuple[List[Dict], List[str], List[str]]:
    """Gold cases, conflicting doc ids, laws missing from the dictionary."""
    tasks = {t.doc_id: t for t in load_tasks(batch)}
    people = annotators(batch)
    labels = {p: load_labels(batch, p) for p in people}
    cases, conflicts, missing = [], [], set()
    for doc_id, task in tasks.items():
        doc_labels = {p: l[doc_id] for p, l in labels.items() if doc_id in l}
        for label in doc_labels.values():
            missing.update(str(r.get("law_name")) for r in label.rows if r.get("law_id") is None)
        if not doc_labels:
            continue
        if ADJUDICATOR in doc_labels:
            chosen, who = doc_labels[ADJUDICATOR], [ADJUDICATOR]
        elif len({frozenset(key_counter(l).items()) for l in doc_labels.values()}) == 1:
            chosen, who = next(iter(doc_labels.values())), sorted(doc_labels)
        else:
            conflicts.append(doc_id)
            continue
        cases.append({
            "id": f"{batch}_{doc_id}",
            "text": task.text,
            "source": task.source,
            "annotators": who,
            "notes": [chosen.comment] if chosen.comment else [],
            "links": label_links(chosen),
        })
    return cases, conflicts, sorted(missing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    for a, b, f1, n in agreement(args.batch):
        print(f"agreement {a} / {b}: F1={f1:.3f} on {n} documents")
    cases, conflicts, missing = export(args.batch)
    total = len(load_tasks(args.batch))
    print(f"exportable: {len(cases)} of {total} documents, {sum(len(c['links']) for c in cases)} links")
    if conflicts:
        print(f"conflicts (label as '{ADJUDICATOR}' to resolve): {', '.join(conflicts)}")
    if missing:
        print(f"laws missing from the dictionary: {'; '.join(missing)}")
    if args.out:
        args.out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1) + "\n", "utf-8")
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

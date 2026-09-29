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


def cached(name: str, factory: Callable[..., object], kw: Dict[str, object]) -> object:
    """Build a linker once per parameter set; `threshold` is applied to the
    cached instance, so threshold sweeps do not retrain anything."""
    kw = dict(kw)
    threshold = kw.pop("threshold", None)
    key = (name, tuple(sorted(kw.items())))
    if key not in _cache:
        _cache[key] = factory(**kw)
    linker = _cache[key]
    if threshold is not None:
        linker.threshold = threshold
    return linker


def tfidf_linker(**kw):
    from research.methods.tfidf import TfidfLinker

    return cached("tfidf", lambda **k: TfidfLinker(DEFAULT_ALIASES_PATH, **k), kw)


def fuzzy_linker(**kw):
    from research.methods.fuzzy import FuzzyLinker

    return cached("fuzzy", lambda **k: FuzzyLinker(DEFAULT_ALIASES_PATH, **k), kw)


def ngram_linker(**kw):
    from research.methods.ngram import NgramLinker

    return cached("ngram", lambda **k: NgramLinker(DEFAULT_ALIASES_PATH, **k), kw)


def w2v_linker(**kw):
    from research.methods.embeddings import Word2VecLinker

    return cached("w2v", lambda **k: Word2VecLinker(DEFAULT_ALIASES_PATH, **k), kw)


BEST_TFIDF = {"threshold": 0.85, "ngram_range": (3, 5)}
BEST_HMM = {"order": 2, "seed": 8, "lambdas": (0.8, 0.15, 0.05)}


def context_step(kind: str):
    from research.methods.context import AnaphoraResolver, ContextNB

    key = ("context", kind)
    if key not in _cache:
        _cache[key] = AnaphoraResolver(DEFAULT_ALIASES_PATH) if kind == "anaphora" else ContextNB(DEFAULT_ALIASES_PATH)
    return _cache[key]


def tagger(kind: str, n_train: int = 20000, seed: int = 7, **kw):
    """Chain tagger trained on synthetic chains in corpus sentences."""
    from research.corpus import training_texts
    from research.methods.hmm import CRFTagger, HMMTagger
    from research.methods.tagging import TaggerChains, plain_sentences, synth_dataset

    key = ("tagger", kind, n_train, seed, tuple(sorted(kw.items())))
    if key not in _cache:
        if "sentences" not in _cache:
            _cache["sentences"] = plain_sentences(training_texts(), 40000)
        data = synth_dataset(_cache["sentences"], n_train, seed=seed)
        model = HMMTagger(**kw) if kind == "hmm" else CRFTagger(**kw)
        _cache[key] = TaggerChains(model.fit(data))
    return _cache[key]


# Each variant takes keyword parameters: "s1_tfidf?threshold=0.5&max_tokens=10".
VARIANTS: Dict[str, Callable[..., object]] = {
    "b0_exact": lambda: Pipeline(
        RegexChains(), ExactMentions.from_json(DEFAULT_ALIASES_PATH), NearestRightLinker()
    ),
    "b1_rules": lambda: Pipeline(RegexChains(), TrieMentions(alias_index()), NearestRightLinker()),
    "s1_tfidf": lambda **kw: Pipeline(RegexChains(), NoMentions(), tfidf_linker(**kw)),
    "s2_fuzzy": lambda **kw: Pipeline(RegexChains(), NoMentions(), fuzzy_linker(**kw)),
    "s3_ngram": lambda **kw: Pipeline(RegexChains(), NoMentions(), ngram_linker(**kw)),
    "s4_w2v": lambda **kw: Pipeline(RegexChains(), NoMentions(), w2v_linker(**kw)),
    # Chain finders, all with the best TF-IDF linker.
    "c_rules": lambda: Pipeline(RegexChains(), NoMentions(), tfidf_linker(**BEST_TFIDF)),
    "c_hmm": lambda **kw: Pipeline(tagger("hmm", **kw), NoMentions(), tfidf_linker(**BEST_TFIDF)),
    "c_crf": lambda **kw: Pipeline(tagger("crf", **kw), NoMentions(), tfidf_linker(**BEST_TFIDF)),
    # Linking of enumerations: chains without a law inherit the next one.
    "p_rules": lambda: Pipeline(RegexChains(), NoMentions(), tfidf_linker(**BEST_TFIDF), propagate=True),
    "p_hmm": lambda **kw: Pipeline(tagger("hmm", **kw), NoMentions(), tfidf_linker(**BEST_TFIDF), propagate=True),
    "x_rules": lambda: Pipeline(
        RegexChains(), NoMentions(), tfidf_linker(**BEST_TFIDF), propagate=True,
        anaphora=context_step("anaphora"), context_nb=context_step("nb"),
    ),
    # HMM2 chains + TF-IDF + context steps (anaphora, naive Bayes for shared law numbers).
    "x_hmm": lambda anaphora=True, nb=True, **kw: Pipeline(
        tagger("hmm", **{**BEST_HMM, **kw}), NoMentions(), tfidf_linker(**BEST_TFIDF), propagate=True,
        anaphora=context_step("anaphora") if anaphora else None,
        context_nb=context_step("nb") if nb else None,
    ),
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

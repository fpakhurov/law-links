import pytest

from law_links import ROOT
from scripts.eval import evaluate, load_cases

CASES = load_cases(ROOT / "tests" / "gold_real.json")
NEGATIVE = [c for c in CASES if not c["links"]]
# Baseline on real texts. Raise it when the extractor improves, never lower it.
F1_FLOOR = 0.82


def test_real_gold_f1_floor(extractor):
    assert evaluate(extractor, CASES)["f1"] >= F1_FLOOR


@pytest.mark.parametrize("case", NEGATIVE, ids=[c["id"] for c in NEGATIVE])
def test_real_negative_case(extractor, case):
    assert extractor.extract(case["text"]) == []

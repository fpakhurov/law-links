import pytest

from law_links import ROOT
from scripts.eval import evaluate, load_cases, to_key

GOLD = ROOT / "tests" / "gold.json"
CASES = load_cases(GOLD)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_gold_case(extractor, case):
    predicted = [to_key(link.model_dump()) for link in extractor.extract(case["text"])]
    expected = [to_key(link) for link in case["links"]]
    assert sorted(predicted, key=str) == sorted(expected, key=str)


def test_order_follows_text(extractor):
    links = extractor.extract("ст. 1 УК РФ и ст. 2 НК РФ")
    assert [(l.law_id, l.article) for l in links] == [(13, "1"), (15, "2")]


def test_gold_f1(extractor):
    assert evaluate(extractor, CASES)["f1"] == 1.0

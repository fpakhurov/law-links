import pytest

from law_links import ROOT
from scripts.eval import evaluate, load_cases, to_key

GOLD = ROOT / "tests" / "gold.json"
CASES = load_cases(GOLD)
# Known disagreements with the gold set, kept visible instead of tuned away.
KNOWN = {
    "neg_sentence_boundary": "'ст. 5. НК РФ': a dot after the article number is read as part of the chain",
}


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=pytest.mark.xfail(reason=KNOWN[c["id"]], strict=True)) if c["id"] in KNOWN else c for c in CASES],
    ids=[c["id"] for c in CASES],
)
def test_gold_case(extractor, case):
    predicted = [to_key(link.model_dump()) for link in extractor.extract(case["text"])]
    expected = [to_key(link) for link in case["links"]]
    assert sorted(predicted, key=str) == sorted(expected, key=str)


def test_order_follows_text(extractor):
    links = extractor.extract("ст. 1 УК РФ и ст. 2 НК РФ")
    assert [(l.law_id, l.article) for l in links] == [(13, "1"), (15, "2")]


def test_enumeration_shares_law(extractor):
    links = extractor.extract("с учетом ч. 6 ст. 15, ст.ст. 64 и 73 УК РФ")
    assert [(l.law_id, l.article, l.point_article) for l in links] == [
        (13, "15", "6"), (13, "64", None), (13, "73", None),
    ]


def test_anaphora(extractor):
    links = extractor.extract("по ч. 1 ст. 19.5 КоАП РФ. Срок по ч. 24 ст. 19.5 названного Кодекса истек.")
    assert [(l.law_id, l.point_article) for l in links] == [(17, "1"), (17, "24")]


def test_shared_law_number_by_context(extractor):
    text = "протокол опроса, проведенного адвокатом в соответствии с подп. 2 п. 3 ст. 6 Закона N 63-ФЗ"
    assert [l.law_id for l in extractor.extract(text)] == [396]


def test_gold_f1(extractor):
    assert evaluate(extractor, CASES)["f1"] >= 0.98

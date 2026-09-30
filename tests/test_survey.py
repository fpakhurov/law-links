import random

from annotation.survey import LevelMarkers, analyze, claim, context, decide, display_span, names_document, wilson, with_law_tail


def test_claim_reading():
    links = [
        {"law_id": 13, "article": "61", "point_article": "1", "subpoint_article": "и"},
        {"law_id": 13, "article": "61", "point_article": "1", "subpoint_article": "к"},
    ]
    assert claim(links, {13: "УК РФ"}, {"S": "п.", "P": "ч."}) == "п. и, к ч. 1 ст. 61 — УК РФ"
    assert claim(links, {13: "УК РФ"}) == "пп. и, к п./ч. 1 ст. 61 — УК РФ"
    assert claim([{"law_id": 0, "article": None, "point_article": "10", "subpoint_article": None}], {0: "АПК РФ"}, {"P": "п."}) == (
        "п. 10 статья не указана — АПК РФ"
    )


def test_level_markers(extractor):
    text = "на основании п. 6 ч. 1 ст. 24.5 КоАП РФ"
    start = text.index("п. 6")
    assert LevelMarkers(extractor, text).within(start, len(text)) == {"S": "п.", "P": "ч."}


def test_names_document():
    assert names_document("ст. 5 УК РФ", 5)
    assert names_document("п. 7.1 Договора поставки", 6)
    assert not names_document("совершены ли они данным лицом (часть 4). Далее", 36)
    assert names_document("согласно п.4.2.1. Договора оплата", 17)

def test_context_cuts_at_whitespace():
    before, fragment, after = context("один два три ст. 5 УК РФ четыре пять", 13, 24, chars=5)
    assert fragment == "ст. 5 УК РФ"
    assert before == "… три " and after == " четыре …"


def test_decide():
    assert decide(["yes", "yes", "no_law"]) == "yes"
    assert decide(["no_law", "no_numbers", "no_law", "unsure"]) == "no_law"
    assert decide(["yes", "no"]) is None
    assert decide(["other_doc", "other_doc", "yes"]) == "other_doc"


def test_wilson_bounds():
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.95 < hi < 0.99


def test_analyze_screens_random_voters_and_recovers_precision():
    rng = random.Random(1)
    items = {}
    truth = {}
    for i in range(200):
        items[f"v{i}"] = {"kind": "verify", "links": [{}], "expected": None}
        truth[f"v{i}"] = "yes" if rng.random() < 0.9 else "no_numbers"
    for i in range(20):
        items[f"c{i}"] = {"kind": "control", "links": [{}], "expected": "yes" if i % 2 else "no"}
    votes = []

    def vote(voter, item, answer):
        votes.append({"vote_id": f"{voter}-{item}", "voter": voter, "item_id": item, "answer": answer, "ms": 3000})

    for g in range(6):  # careful voters, 5% slips
        for item_id, item in items.items():
            right = truth.get(item_id) or ("yes" if item["expected"] == "yes" else "no_law")
            vote(f"good{g}", item_id, right if rng.random() > 0.05 else rng.choice(["yes", "no_law"]))
    for r in range(2):  # random clickers
        for item_id in items:
            vote(f"rand{r}", item_id, rng.choice(["yes", "no_law", "no_numbers", "no_ref"]))

    result = analyze(items, votes)
    assert result["kept"] == 6
    assert not result["voters"]["rand0"]["kept"] and not result["voters"]["rand1"]["kept"]
    true_precision = sum(t == "yes" for t in truth.values()) / len(truth)
    assert abs(result["precision"] - true_precision) < 0.02
    lo, hi = result["precision_ci"]
    assert lo < true_precision < hi


def test_with_law_tail():
    text = "положений ст. 15 УПК РФ, в условиях"
    assert text[: with_law_tail(text, text.index(" РФ"))].endswith("УПК РФ")
    assert with_law_tail("ст. 15 УПК Российской Федерации.", 10) == len("ст. 15 УПК Российской Федерации")
    assert with_law_tail("ст. 15 УПК РФы", 10) == 10


def test_display_span_stops_before_next_chain():
    text = "с учетом ч. 6 ст. 15, ст.ст. 64 и 73 УК РФ суд"
    start, nxt = text.index("ч. 6"), text.index("ст.ст.")
    assert text[start : display_span(text, start, text.index(" суд"), [start, nxt])] == "ч. 6 ст. 15"
    assert text[nxt : display_span(text, nxt, text.index(" РФ"), [start, nxt])] == "ст.ст. 64 и 73 УК РФ"

import pytest

import annotation.store as store
from annotation.export import agreement, export
from annotation.laws import choice, law_titles, parse_choice
from annotation.store import Label, Task, expand_row, find_all_fragments, find_fragment, save_label, split_values, validate_row, write_tasks
from annotation.suggest import suggestion_rows


def row(**kw):
    base = {"fragment": "ст. 3 НК РФ", "law_id": 15, "law_name": "", "article": "3", "point": "", "subpoint": ""}
    return {**base, **kw}


def test_split_values():
    assert split_values("1, 2 ,3") == ["1", "2", "3"]
    assert split_values("43.2 - 6") == ["43.2-6"]
    assert split_values("") == [None]


def test_expand_row_product():
    links = expand_row(row(article="3", point="2", subpoint="1, 2"))
    assert [(l["point_article"], l["subpoint_article"]) for l in links] == [("2", "1"), ("2", "2")]


def test_unknown_law_gives_no_links():
    assert expand_row(row(law_id=None, law_name="Конституция РФ")) == []


def test_find_fragment_tolerates_whitespace():
    assert find_fragment("по  ст.\n3 НК РФ", "ст. 3 НК РФ") == (4, 15)
    assert find_fragment("текст", "ст. 3") is None
    assert find_all_fragments("ст. 3 и ст.  3", "ст. 3") == [(0, 5), (8, 14)]


def test_validate_row():
    assert validate_row("Согласно ст. 3 НК РФ", row()) == []
    assert len(validate_row("Согласно ст. 3 НК РФ", row(fragment="ст. 4", article="", law_id=None))) == 3


def test_law_titles_tell_shared_numbers_apart():
    titles = law_titles({
        "246": ["ФЗ №63-ФЗ", "Федеральный закон №63-ФЗ «Об электронной подписи»"],
        "396": ["ФЗ №63-ФЗ", "Федеральный закон №63-ФЗ «Об адвокатской деятельности и адвокатуре в Российской Федерации»"],
    })
    assert "электронной" in titles[246] and "адвокатской" in titles[396]
    assert parse_choice(choice(396, titles[396])) == 396


def test_suggestion_rows_merge_product(extractor):
    rows = suggestion_rows(extractor, "пп. 1, 2 п. 2 ст. 3 НК РФ")
    assert len(rows) == 1
    assert (rows[0]["article"], rows[0]["point"], rows[0]["subpoint"]) == ("3", "2", "1, 2")


@pytest.fixture
def batch_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "TASKS_DIR", tmp_path / "tasks")
    monkeypatch.setattr(store, "LABELS_DIR", tmp_path / "labels")
    write_tasks("b", [Task("d1", "ст. 3 НК РФ", "s1"), Task("d2", "ст. 5 УК РФ", "s2")])
    return "b"


def test_export_agreement_and_conflicts(batch_dir):
    save_label(batch_dir, "ann1", Label("d1", rows=[row()]))
    save_label(batch_dir, "ann2", Label("d1", rows=[row()]))
    save_label(batch_dir, "ann1", Label("d2", rows=[row(fragment="ст. 5 УК РФ", law_id=13, article="5")]))
    save_label(batch_dir, "ann2", Label("d2", rows=[row(fragment="ст. 5 УК РФ", law_id=13, article="6")]))
    [(a, b, f1, n)] = agreement(batch_dir)
    assert (a, b, n) == ("ann1", "ann2", 2) and f1 == pytest.approx(0.5)
    cases, conflicts, _ = export(batch_dir)
    assert [c["id"] for c in cases] == ["b_d1"] and conflicts == ["d2"]
    save_label(batch_dir, "adjudicator", Label("d2", rows=[row(fragment="ст. 5 УК РФ", law_id=13, article="5")]))
    cases, conflicts, _ = export(batch_dir)
    assert len(cases) == 2 and not conflicts


def test_last_save_wins(batch_dir):
    save_label(batch_dir, "ann1", Label("d1", rows=[row()]))
    save_label(batch_dir, "ann1", Label("d1", no_links=True))
    cases, _, _ = export(batch_dir)
    assert cases[0]["links"] == []


def test_invalid_annotator_name(batch_dir):
    with pytest.raises(ValueError):
        save_label(batch_dir, "../x", Label("d1"))

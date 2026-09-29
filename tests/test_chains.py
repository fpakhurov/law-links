import pytest

from law_links.chains import tokenize, value_text


def chain(chains, text):
    found = chains.find(text)
    assert len(found) == 1, found
    c = found[0]
    return c.articles, c.points, c.subpoints


@pytest.mark.parametrize(
    "text, expected",
    [
        ("пп. 1 п. 1 ст. 374", (["374"], ["1"], ["1"])),
        ("подпункту 2 пунта б статьи 22", (["22"], ["б"], ["2"])),
        ("в подпунктах а, б и с пункта 3.345, 23 в статье 66", (["66"], ["3.345", "23"], ["а", "б", "с"])),
        ("часть 3, ст. 30.1", (["30.1"], ["3"], [])),
        ("Согласно п.  10 АПК", ([], ["10"], [])),
        ("ст. 30 и в соответствии", (["30"], [], [])),
        # constructions from court decisions
        ("по ч. 1 ст. 19.5. КоАП РФ", (["19.5"], ["1"], [])),
        ("в силу п. «б» ч. 1 ст. 58 УК РФ", (["58"], ["1"], ["б"])),
        ("на основании п. 6 ч. 1 ст. 24.5 КоАП РФ", (["24.5"], ["1"], ["6"])),
        ('подпунктом "г" пункта 6 части 1 статьи 81 Трудового кодекса', (["81"], ["6"], ["г"])),
        pytest.param(
            "пункту 1 части первой статьи 81 ТК РФ", (["81"], ["1"], ["1"]),
            marks=pytest.mark.xfail(reason="known: point before an ordinal part is lost", strict=True),
        ),
        ("руководствуясь ст.ст. 194 -199 ГПК РФ", (["194-199"], [], [])),
        ("согласно абз. 1 ст. 394 ТК РФ", (["394"], [], [])),
        ("как ч.1, так и ч.24 ст.19.5 КоАП РФ", (["19.5"], ["1", "24"], [])),
    ],
)
def test_chain(chains, text, expected):
    assert chain(chains, text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "пост 5 и степ 6",
        "характеризуется удовлетворительно (л.д. 55-61), что",
        "по адресу: ст. Отрадная, ул. Широкая",
        "по правилам соответствующей главы части второй Кодекса",
    ],
)
def test_no_chain(chains, text):
    assert chains.find(text) == []




def test_value_text():
    assert value_text("43.2 - 6") == "43.2-6"
    assert value_text("первой") == "1"
    assert [t.text for t in tokenize("ст.ст. 19.5.")] == ["ст", ".", "ст", ".", "19.5", "."]

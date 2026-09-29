import pytest

from law_links.grammar import find_chains, split_values


@pytest.mark.parametrize(
    "group, expected",
    [
        ("1", ["1"]),
        ("4.6", ["4.6"]),
        ("43.2-6", ["43.2-6"]),
        ("43.2 - 6", ["43.2-6"]),
        ("1, 2, 3", ["1", "2", "3"]),
        ("5 и 6", ["5", "6"]),
        ("4, 5, 6 и 8", ["4", "5", "6", "8"]),
        ("а, б и с", ["а", "б", "с"]),
        ("", []),
    ],
)
def test_split_values(group, expected):
    assert split_values(group) == expected


def chain(text):
    chains = find_chains(text)
    assert len(chains) == 1, chains
    c = chains[0]
    return c.articles, c.points, c.subpoints


def test_full_chain():
    assert chain("пп. 1 п. 1 ст. 374") == (["374"], ["1"], ["1"])


def test_words_and_typo():
    assert chain("подпункту 2 пунта б статьи 22") == (["22"], ["б"], ["2"])


def test_preposition_between_levels():
    assert chain("в подпунктах а, б и с пункта 3.345, 23 в статье 66") == (
        ["66"],
        ["3.345", "23"],
        ["а", "б", "с"],
    )


def test_part_with_comma():
    assert chain("часть 3, ст. 30.1") == (["30.1"], ["3"], [])


def test_point_without_article():
    assert chain("п.  10") == ([], ["10"], [])


def test_preposition_not_taken_as_value():
    assert chain("ст. 30 и в соответствии") == (["30"], [], [])


def test_no_marker_inside_word():
    assert find_chains("пост 5 и степ 6") == []

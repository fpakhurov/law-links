from law_links.chains import Chain
from law_links.normalize import normalize


def test_normalize_preserves_length():
    text = "«Закон» —\tё"
    assert len(normalize(text)) == len(text)


def test_normalize_folds_homoglyphs_in_mixed_words():
    assert normalize("cт. 5 УK PФ, Latin") == "ст. 5 УК РФ, Latin"


def test_lookup_nominative(resolver):
    assert resolver.lookup("Налоговый кодекс РФ") == [15]


def resolve(resolver, text):
    """Law of a chain that ends right before `text`."""
    mention = resolver.resolve(normalize(text), Chain(0, 0, ["1"], [], []))
    return mention.law_id if mention else None


def test_declension(resolver):
    assert resolve(resolver, " Гражданского процессуального кодекса Российской Федерации") == 6


def test_abbreviation_without_rf(resolver):
    assert resolve(resolver, " КоАП, суд") == 17


def test_old_title_and_latin_n(resolver):
    text = " Федерального закона от 01.04.1996 N 27-ФЗ «Об индивидуальном (персонифицированном) учете в системе обязательного пенсионного страхования»"
    assert resolve(resolver, text) == 498


def test_not_a_law(resolver):
    assert resolve(resolver, " Договора поставки") is None


def test_stops_at_sentence_end(resolver):
    assert resolve(resolver, " настоящего решения. НК РФ") is None


def test_typos_in_law_name(resolver):
    assert resolve(resolver, " Трудовго коедкса Российской Федерации") == 10
    assert resolver.spell("часто") == "часто"  # a known word is never corrected

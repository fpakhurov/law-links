from law_links.normalize import normalize


def test_normalize_preserves_length():
    text = "«Закон» —\tё"
    assert len(normalize(text)) == len(text)


def test_lookup_nominative(index):
    assert index.lookup("Налоговый кодекс РФ") == [15]


def test_mentions_declension(index):
    mentions = index.find_mentions(normalize("по статье Гражданского процессуального кодекса"))
    assert [m.law_id for m in mentions] == [6]


def test_longest_match_prefers_full_name(index):
    text = normalize('Федерального закона "О государственной регистрации недвижимости"')
    mentions = index.find_mentions(text)
    assert [m.law_id for m in mentions] == [174]
    assert mentions[0].start == 0


def test_regular_word_not_taken_as_abbreviation(index):
    assert index.find_mentions("как") == []


def test_no_match_across_sentence(index):
    assert [m.law_id for m in index.find_mentions("НК. России")] == [15]

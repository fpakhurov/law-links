from research.ner_corpus import anonymize, clean


def test_clean_collapses_layout_whitespace():
    assert clean("\r\n\t  Дело  № 5\r\n\n граж­да­нин  ") == "Дело № 5\nгражданин"


def test_anonymize_masks_names_in_every_form():
    text = (
        "в отношении Насртдиновой ДТ. Гражданка Насртдинова Д.Т., ТУРБИНА Виктора Сергеевича, "
        "судья О.Н. Стасенко, Гузь А., мировой судья по ст. 12.8 ч. 1 КоАП РФ, Кодекса РФ"
    )
    assert anonymize(text) == (
        "в отношении ФИО ДТ. Гражданка ФИО, ФИО, "
        "судья ФИО, ФИО, мировой судья по ст. 12.8 ч. 1 КоАП РФ, Кодекса РФ"
    )


def test_anonymize_stays_within_a_line():
    assert anonymize("Иванов И.И.\nО.А.\nРассмотрев дело") == "ФИО\nО.А.\nРассмотрев дело"

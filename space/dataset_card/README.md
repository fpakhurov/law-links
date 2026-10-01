---
license: cc-by-4.0
language:
- ru
pretty_name: Law Links Survey Votes
task_categories:
- token-classification
tags:
- legal
- russian
- human-feedback
size_categories:
- n<1K
---

# Law Links: ответы опроса о юридических ссылках

Ответы «да/нет» на вопросы о ссылках на нормы законов в текстах российских судебных решений: верно ли программа нашла и поняла ссылку («ч. 3 ст. 158 УК РФ» -> статья 158, часть 3, Уголовный кодекс) и есть ли ссылка в фрагментах, которые программа пропустила. - Опрос: [fpakhurov/law-links-survey](https://huggingface.co/spaces/fpakhurov/law-links-survey)
- Код сервиса, опроса и анализа: [github.com/fpakhurov/law-links](https://github.com/fpakhurov/law-links)
- Методика: [docs/SURVEY.md](https://github.com/fpakhurov/law-links/blob/master/docs/SURVEY.md), технический отчёт: [REPORT.md](https://github.com/fpakhurov/law-links/blob/master/REPORT.md)
- Анализ ответов: `python -m annotation.survey analyze survey1 --votes <папка с votes>`, так же `survey2` (в репозитории)

## Файлы

- `items/survey1.jsonl`, `items/survey2.jsonl` - вопросы двух пакетов (в `survey2` своих контрольных нет, используются контрольные `survey1`; поле `pack` - имя пакета). Поля: `item_id`, `kind` (`verify` - проверка найденной ссылки, `missed` - кандидат на пропуск, `control` - вопрос с известным ответом), `doc_id`, `source` (ссылка на решение на sudact.ru и строки), `before` / `fragment` / `after` (контекст, выделенный фрагмент), `claim` (как программа поняла фрагмент, записано как ссылка: `п. 6 ч. 1 ст. 24.5 — КоАП РФ`), `fields` (то же по элементам: `law`, `article`, `lower`), `links` (ссылки в формате `{law_id, article, point_article, subpoint_article}`, `law_id` по словарю задания), `expected` (ответ для `control`).
- `votes/votes-*.jsonl` - ответы, дописываются во время опроса. Поля: `vote_id`, `item_id`, `kind`, `answer` (для `verify`/`control`: `yes`, `no_law` (неверный закон), `no_article` (неверная статья), `no_part` (неверная часть, пункт или подпункт), `no_ref` (не ссылка), `unsure` (пропуск); в первых ответах вместо `no_article`/`no_part` было общее `no_numbers`; для `missed`: `yes` - ссылка на закон или кодекс, `other_doc` - на другой документ, `no`, `unsure`), `voter` (случайный идентификатор браузера), `name` (придуманное участником имя), `ms` (время на ответ), `time` (UTC).

## Происхождение и ограничения

Тексты - фрагменты судебных актов с sudact.ru, обезличенные источником; судебные акты не охраняются авторским правом (п. 6 ст. 1259 ГК РФ). Вопросы `survey1` собраны моделью law-links v1.0.1, `survey2` - v1.2.1, по разным документам; приложение показывает оба пакета вместе. Отвечали студенты, не юристы; невнимательные участники отсеиваются по контрольным вопросам (`python -m annotation.survey analyze`). Полнота, оценённая по вопросам `missed`, относится только к пулу кандидатов, а не ко всем ссылкам в текстах.

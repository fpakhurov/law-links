# Опрос: проверка найденных ссылок

Полной разметки не будет, поэтому качество оценивается ответами «да/нет» одногруппников на заранее собранные вопросы. Приложение опроса (`space/`) работает на Hugging Face Space, ответы уходят в открытый датасет [fpakhurov/law-links-votes](https://huggingface.co/datasets/fpakhurov/law-links-votes) (CC-BY-4.0), участники предупреждены об этом во вступлении.

## Что спрашиваем

| Вид | Что видит участник | Ответы | Что даёт |
|---|---|---|---|
| verify | найденный фрагмент в контексте и его прочтение: «Статья 158, часть/пункт 3 — Уголовный кодекс РФ» | верно / не тот закон / не те номера / это не ссылка / не понять | точность и типы ошибок |
| missed | кандидат, которого модель не вернула: цепочка без закона или маркер с номером вне цепочек | да / нет / не понять | полнота относительно пула кандидатов |
| control | как verify, ответ известен: верное прочтение из dev или испорченное (другой номер статьи, другой кодекс) | как verify | отсев невнимательных участников |

Пакет `survey1`: 40 свежих документов корпуса, 273 verify, 90 missed, 40 control (`python -m annotation.survey build survey1`). Каждому участнику сначала показываются вопросы с наименьшим числом голосов, каждый 8-й вопрос контрольный, цель - 2 голоса на вопрос (~730 ответов).

Ограничение: полнота считается только относительно пула кандидатов. Ссылка, которую не нашла ни модель, ни поиск маркеров, в опрос не попадает.

## Развёртывание

1. Открытый датасет для ответов: `hf repos create fpakhurov/law-links-votes --repo-type dataset`, карточка и вопросы: `hf upload fpakhurov/law-links-votes space/dataset_card . --repo-type dataset`.
2. Space: `hf repos create fpakhurov/law-links-survey --repo-type space --space-sdk gradio` (публичный, иначе одногруппники не откроют).
3. В настройках Space (Settings -> Variables and secrets):
   - секрет `HF_TOKEN`: fine-grained токен с правом записи только в `fpakhurov/law-links-votes` (https://huggingface.co/settings/tokens);
   - переменная `VOTES_REPO` = `fpakhurov/law-links-votes`.
4. Загрузка приложения: `python -m annotation.survey space survey1`, затем `hf upload fpakhurov/law-links-survey space . --repo-type space --exclude "votes/*" --exclude "dataset_card/*"`.
5. Ссылка для одногруппников: https://huggingface.co/spaces/fpakhurov/law-links-survey

Бесплатный Space засыпает после 48 часов без посетителей, первый заход после этого ждёт запуска около минуты. Ответы отправляются в датасет раз в 2 минуты: при перезапуске Space теряются ответы последних 2 минут.

## Анализ

```bash
hf download fpakhurov/law-links-votes --repo-type dataset --local-dir annotation/data/survey/survey1/votes
python -m annotation.survey analyze survey1 --votes annotation/data/survey/survey1/votes
```

Участник учитывается, если ответил хотя бы на 3 контрольных вопроса и верно на 75% из них. Решение по вопросу - большинство учтённых голосов без «не понять», ничья не решена. Отчёт пишется в `annotation/data/survey/survey1/report.md`, решения по вопросам в `labels.jsonl` (данные для верификатора и дообучения HMM на подтверждённых цепочках).

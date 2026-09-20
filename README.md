# Bot detection

Решение тестового задания по детекции автоматизированного трафика.

Целевая метрика: максимальный Precision при Recall >= 70%.

## Что делала

Сначала посмотрела данные, распределение target, возраст cookie, события и интервалы между ними.

На первом baseline результат был около 0.39 по P@R>=70%.

Дальше добавила признаки по событиям и поведению cookie:
- количество и доли разных событий
- возраст cookie
- gap между событиями
- активность по минутам и часам
- категории, локации, поиск
- User-Agent
- дубли
- entropy и top share
- переходы между событиями

После этого качество выросло примерно до 0.63 на последнем временном фолде.

Потом расширила обычные признаки и отдельно добавила cross-cookie признаки.

Для `user_agent`, `item_id`, `search_query`, `item_category` и `item_location` считаю, насколько часто такие значения встречались у других cookie.

Здесь отдельно контролирую время. Для каждой cookie в reference попадают только события с `event_ts < window_end_ts` этой cookie. Для train также используется leave-one-cookie-out, чтобы собственная cookie не увеличивала частоту своих значений.

## Валидация

Train и test разделены по времени, поэтому случайный split не использовала.

Проверяла решение на трех последовательных временных фолдах:
- 13-14 апреля
- 15-16 апреля
- 17-19 апреля

На расширенном наборе обычных признаков средний P@R>=70% получился около 0.76.

После добавления temporal cross-cookie признаков:

| Модель | Mean P@R>=70% |
|---|---:|
| CatBoost | 0.822 |
| HistGradientBoosting | 0.840 |
| Rank blend | 0.844 |

Для rank blend по трем фолдам получилось примерно:
- 0.806
- 0.889
- 0.838

Лучший средний результат дал rank blend.

## Финальная модель

Использую две модели:
- `HistGradientBoostingClassifier`
- `CatBoostClassifier`

Финальный score считаю как rank blend:

```text
0.55 * rank(HistGradientBoosting) + 0.45 * rank(CatBoost)
```

В итоговом наборе около 250 признаков.

## Структура проекта

```text
bot_detection_case/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── events.csv.gz
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   ├── 03_features.ipynb
│   ├── 04_cross_cookie.ipynb
│   └── 05_final.ipynb
│
├── src/
│   ├── features_base.py
│   └── features_cross.py
│
├── metric.py
├── sample_submission.csv
├── submission.csv
├── requirements.txt
└── README.md
```

## Как запустить

Python 3.12.14

Установка зависимостей:

```bash
pip install -r requirements.txt
```

Основные библиотеки:
- NumPy
- pandas
- scikit-learn
- CatBoost

Точные версии указаны в `requirements.txt`.

Для воспроизведения финального результата достаточно запустить:

```text
notebooks/05_final.ipynb
```

Ноутбук:
1. загружает train, test и events
2. фильтрует события по индивидуальным окнам
3. строит обычные признаки
4. строит temporal cross-cookie признаки
5. обучает HistGradientBoosting и CatBoost
6. считает rank blend
7. сохраняет `submission.csv`

## Submission

Финальный файл содержит две колонки:

```text
cookie_id,score
```

Для каждой test cookie есть одна строка.

`score` непрерывный и лежит в диапазоне от 0 до 1. Порог заранее не выбирается.

В моделях зафиксирован `random_state = 2026`.

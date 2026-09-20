# Bot detection

Решение тестового задания по детекции ботов
Метрика: Precision при Recall >= 70%

## Что делала

Сначала посмотрела данные и собрала простой baseline. Он дал P@R>=70% около 0.389
Дальше добавила признаки по событиям, времени между событиями, категориям, локациям, поиску и User-Agent. CatBoost дошел примерно до 0.57
После более полного feature engineering результат на последнем временном фолде вырос примерно до 0.635
Самый большой прирост дали cross-cookie признаки. Для нескольких полей я считаю, насколько часто такое же значение встречалось у других cookie:

- user_agent
- item_id
- search_query
- item_category
- item_location

Для validation эти частоты считаются только по более раннему train. Для train используется leave-one-cookie-out. Так validation не попадает в признаки

## Валидация

Train заканчивается 19 апреля, test начинается 20 апреля. Поэтому случайный split не использовала

Проверяла решение на трех временных фолдах:

- 13-14 апреля
- 15-16 апреля
- 17-19 апреля

Итоговый rank blend дал средний P@R>=70% около 0.886 на трех временных фолдах

## Финальная модель

Использую:

- HistGradientBoostingClassifier
- CatBoostClassifier
- rank blend 0.55 / 0.45

В финальном наборе около 250 признаков

`05_final.ipynb` обучает модели на всем train и сохраняет `submission.csv`

## Как запустить

Python 3.12.14

```bash
pip install -r requirements.txt
```

После установки зависимостей запустить:

```text
notebooks/05_final.ipynb
```

Ожидаемая структура данных:

```text
data/
  train.csv
  test.csv
  events.csv.gz
```

## Файлы

```text
notebooks/
  01_eda.ipynb
  02_modeling.ipynb
  03_features.ipynb
  04_cross_cookie.ipynb
  05_final.ipynb

src/
  features_base.py
  features_cross.py

metric.py
requirements.txt
README.md
submission.csv
```

Во всех финальных моделях random_state = 2026

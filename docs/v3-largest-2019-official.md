# Отдельный официальный LargeST-CA 2019 / 15 min

Этот эксперимент изолирован от локального `LargeST-CA-2017 / 5 min` и не
перезаписывает его данные, конфигурацию или артефакты. Числа этих двух протоколов
нельзя объединять в одну строку таблицы.

## Контракт данных

Источник — официальный `ca_his_raw_2019.h5` из набора LargeST. Официальный
notebook `data/ca/process_ca_his.ipynb` выполняет ровно:

```python
ca_his.resample("15T").mean().round(0).fillna(0)
```

Подготовленный файл хранится отдельно в
`data/LargeST/official_2019_15min/ca_his_2019.h5`. Скрипт записывает SHA-256
исходного и выходного файлов и отказывается перезаписывать результат без
`--force`.

```powershell
.\.venv\Scripts\python.exe .\scripts\v3\prepare_largest_2019.py `
  --source data\LargeST\ca_his_raw_2019.h5 `
  --output data\LargeST\official_2019_15min\ca_his_2019.h5
```

Конфигурация воспроизводит официальный sample protocol: 12 точек истории,
12 точек прогноза, 15 минут на точку и split 60/20/20 по всем допустимым
sliding-window anchors с Python `round`, а не по границам исходного timeline.
Это даёт горизонты 15, 45, 90 и 180 минут в common evaluator; основные
сопоставимые с таблицей LargeST горизонты — 45/90/180 минут.

## Проверка и будущий запуск

```powershell
.\.venv\Scripts\python.exe -m dagnn_pistnet_v3 inspect-data `
  --config configs\v3\large_st_ca_2019_15min_official.yaml

.\.venv\Scripts\python.exe -m dagnn_pistnet_v3 export-protocol `
  --config configs\v3\large_st_ca_2019_15min_official.yaml `
  --output artifacts\v3\protocols\large_st_ca_2019_15min_official.json

.\.venv\Scripts\python.exe -m dagnn_pistnet_v3 study `
  --spec configs\v3\publication_study_largest_2019_official.yaml --dry-run

.\.venv\Scripts\python.exe -m dagnn_pistnet_v3 study `
  --spec configs\v3\publication_study_largest_2019_official.yaml
```

Последнюю команду следует поставить в последовательную GPU-очередь после
текущего исследования; параллельный тяжёлый запуск на той же GPU не является
сопоставимым по wall time и повышает риск OOM.

## Современный large-scale comparator

Для отдельной LargeST-таблицы основной современный масштабируемый reference —
[FaST (KDD 2026)](https://github.com/yijizhao/FaST). Его официальный репозиторий
использует `ca_his_raw_2019.h5`, sliding windows, split 6:2:2 и сообщает
эксперименты на RTX A6000 48 GB. Но опубликованные FaST-конфиги имеют другие
длины входа/горизонта. Поэтому локальный confirmatory comparator должен получить
новый matched config `12 -> 12`, использовать ровно protocol JSON выше и
вернуть predictions в common evaluator. Опубликованные числа FaST нельзя
подставлять вместо такого прогона.

План и незаполненные provenance-поля находятся в
`configs/v3/large_st_2019_competitors.template.yaml`. До закрепления commit,
создания environment lock и CPU bridge smoke test GPU-запуск не разрешён.
STAEformer и PDFormer тоже пробуются на том же протоколе, но их OOM/failed runs
на 8 600 узлах сохраняются как результат, а не скрываются и не заменяются
числами из другого протокола.

## Ограничение physics/event claims

Официальный LargeST-CA содержит только поток. В нём нет независимо измеренной
плотности, segment volume/boundary flow и event labels. Поэтому в этом конфиге
`use_hard_conservation: false` и `use_event_jump: false`. Эксперимент валидирует
точность прогнозирования flow и масштабируемость, но **не валидирует**
physics/conservation или event-response модули. Такие утверждения требуют
отдельного датасета с соответствующими наблюдаемыми величинами.

Официальные источники:

- <https://github.com/liuxu77/LargeST>
- <https://github.com/liuxu77/LargeST/blob/main/data/ca/process_ca_his.ipynb>
- <https://github.com/liuxu77/LargeST/blob/main/data/generate_data_for_training.py>

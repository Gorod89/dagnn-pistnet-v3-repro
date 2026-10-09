# Предрегистрационный протокол DAGNN + PI-STNet v3

## Основной вопрос

Уменьшает ли полный v3 MAE на 60 мин относительно v2 и сильнейшего локально
перезапущенного конкурента при одинаковых raw data, split, scaler, mask и seeds?

## Первичная конечная точка

- метрика: MAE;
- горизонт: 60 минут;
- анализ: paired difference по одинаковым test windows;
- uncertainty: 95% paired moving-block bootstrap CI (`block_length=12` окон);
- критерий подтверждения: верхняя граница paired CI ниже нуля и
  Holm-adjusted p-value ниже 0,05; обычный CI не называется Holm-corrected;
- датасеты подтверждения: METR-LA, PEMS-BAY, PEMS07, PEMS08;
- LargeST анализируется отдельно из-за node partitioning и масштаба.

## Вторичные точки

- MAE/RMSE/WAPE/sMAPE на 5/15/30 мин;
- CRPS, NLL, coverage и interval width;
- параметры, latency, throughput, peak VRAM и wall time;
- rollout 120/240 мин как stress test, не как основной benchmark;
- robustness к missing sensors, topology dropout и доступным incidents.

## Seeds и model selection

Seeds: `13, 42, 97`. Early stopping использует только validation. Test вызывается
один раз после загрузки лучшего validation checkpoint. Изменение архитектуры или
гиперпараметров после просмотра test создаёт новую, явно названную confirmatory
study; старые результаты не удаляются.

## Абляции

- `no_delta`;
- `no_hierarchy`;
- `no_operator_splitting`;
- `no_probabilistic_head`;
- `no_rollout_curriculum`;
- `no_cross_dataset_pretraining`;
- `no_decomposition`;
- event jump и hard conservation — отдельный protocol при наличии валидных данных.

## Fairness

- одинаковые test targets и common evaluator;
- одинаковое число seeds;
- одинаковый hyperparameter-search budget на семейство модели;
- опубликованные числа не смешиваются с локальными;
- failures/OOM учитываются, а не исключаются после просмотра результатов;
- лучший конкурент выбирается до confirmatory test по validation или литературе.

## Ограничение утверждений

Без incident data нельзя утверждать эффективность jump-модуля. Без независимых
density/flow, segment volume и boundary flow нельзя утверждать физическую
согласованность на основании пяти текущих датасетов.

Текущие flow-only/speed-only benchmarks не валидируют физическую корректность
physics-компонент или причинную эффективность event/jump-модулей. Эти компоненты
считаются архитектурной возможностью до отдельной проверки на physics-ready и
incident-annotated данных; соответствующие claims по текущим датасетам не делаются.

Из-за перекрытия соседних test windows основная confirmatory inference использует
circular moving-block bootstrap и block sign-flip: временной блок, а не отдельное
окно, является единицей ресэмплинга/перестановки. Блоки не пересекают границы
seed. Для 12-step окон заранее фиксируется `block_length=12`; IID window-анализ
публикуется лишь как sensitivity check.

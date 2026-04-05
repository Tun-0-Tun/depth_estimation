# Конфиги экспериментов

Запуск: `uv run python -m scripts.run_experiment --config configs/<имя>.json`.

**Вместо удалённых `eval_depth_anything_nyu.py` / `eval_local_calibration_nyu.py`:**  
`exp_global_nyu.json` (только global), `exp_global_local_nyu.json` (global + local bilateral), либо любой свой набор методов в `methods`.

## Сравнение локальной калибровки: без сглаживания / Гаусс / билатеральное

Один прогон на **20 кадрах NYU** с общим sparse prior (30%) и одной загрузкой Depth Anything на кадр:

```bash
cd /path/to/depth_estimation
uv run python -m scripts.run_experiment --config configs/exp_local_smoothing_nyu20.json
```

Результаты:

- `outputs/experiments/local_smoothing_nyu20/config.json` — зафиксированные параметры
- `outputs/experiments/local_smoothing_nyu20/results.json` — метрики по кадрам и средние
- `outputs/experiments/local_smoothing_nyu20/nyu_*.png` — визуализация (RGB, суперпиксели, все методы, GT)

**Методы в этом конфиге:**

| Ключ в `results.json` | Смысл |
|----------------------|--------|
| `global` | Одна аффинная калибровка на кадр |
| `local_none` | SLIC + локальный (s,t), **без** сглаживания полей |
| `local_gaussian` | то же + **Гаусс** по полям s, t (`sigma=15`) |
| `local_bilateral` | то же + **билатеральное** сглаживание (s,t): на границе суперпикселей с похожими (s,t) граница размывается, при сильном скачке — остаётся резкой |

Параметры билатераля в JSON: `sigma_spatial`, `range_scale` (множитель к `std(s)`, `std(t)` для диапазонного ядра), опционально явные `sigma_range_s` / `sigma_range_t`, `bilateral_max_radius` (потолок окна, память).

## Сравнение нескольких экспериментов

После нескольких прогонов:

```bash
uv run python scripts/compare_experiments.py \
  outputs/experiments/local_smoothing_nyu20/results.json \
  outputs/experiments/baseline/results.json
```

Экспорт в CSV:

```bash
uv run python scripts/compare_experiments.py \
  outputs/experiments/local_smoothing_nyu20/results.json \
  --csv outputs/compare_smoothing.csv
```

## Быстрый режим без JSON

Один метод `local` с билатералем:

```bash
uv run python -m scripts.run_experiment \
  --name bilateral_quick \
  --methods local \
  --num-samples 5 \
  --smooth-mode bilateral \
  --sigma-spatial 6 \
  --range-scale 0.25
```

## INR (GPU): простой MLP и FiLM по регионам

Методы **`inr_simple`** и **`inr_film`** на каждом кадре обучаются на разрежённом prior (как в остальных экспериментах), инференс — плотная карта глубины. Нужен GPU/MPS/CPU с PyTorch.

```bash
uv run python -m scripts.run_experiment --config configs/exp_inr_compare.json
```

Параметры: `train_steps`, `lr`, `hidden_dim`, `num_frequencies`, для FiLM — `n_segments`, `crop_size`, `d_c`, `num_film_layers`. `affine_baseline: true` — предсказание остатка к глобальному \(s\,d_\text{rel}+t\). `train_seed` в конфиге + индекс кадра задают `torch.manual_seed` для воспроизводимости.

Результаты: `outputs/experiments/inr_compare_nyu10/`.

## Визуализация: карты ошибки pred − GT

В `outputs/experiments/.../nyu_*.png` по умолчанию **второй ряд** — знаковая ошибка в метрах (`coolwarm`, общий симметричный лимит по 98-му перцентилю |ошибки| по всем методам на кадре). Отключить: в JSON эксперимента `"show_prediction_diff": false` или в CLI быстрого режима флаг `--no-diff`.

## Сетка числа суперпикселей (`local` без сглаживания)

Та же сетка **`n_segments`:** 50, 100, 150, 200, 300, 400, что у `exp_bilateral_segments_grid.json`, но **`smooth_mode: none`** (кусочно-постоянные s, t по SLIC, без пост-обработки полей).

```bash
uv run python -m scripts.run_experiment --config configs/exp_local_none_segments_grid.json
```

Результаты: `outputs/experiments/local_none_segments_grid_nyu20/`.

## Сетка числа суперпикселей (только `local_bilateral`)

Фиксированные параметры билатераля как в `exp_local_smoothing_nyu20` (`sigma_spatial=6`, `range_scale=0.25`).  
**Сетка `n_segments`:** 50, 100, 150, 200, 300, 400 — от грубого разбиения до более мелких регионов (больше границ, меньше пикселей на суперпиксель при том же prior).

```bash
uv run python -m scripts.run_experiment --config configs/exp_bilateral_segments_grid.json
```

Результаты: `outputs/experiments/bilateral_segments_grid_nyu20/`.  
Сравнение с прошлым экспериментом:

```bash
uv run python scripts/compare_experiments.py \
  outputs/experiments/bilateral_segments_grid_nyu20/results.json \
  outputs/experiments/local_smoothing_nyu20/results.json
```

## Высокое число суперпикселей (400–700, `local_bilateral`)

Те же параметры билатераля, сетка **`n_segments`:** 400, 500, 600, 700 плюс `global`.

```bash
uv run python -m scripts.run_experiment --config configs/exp_bilateral_segments_400_700.json
```

Результаты: `outputs/experiments/bilateral_segments_400_700_nyu20/`.

Другие конфиги: `baseline.json`, `smooth_sweep.json`, `segments_sweep.json`.

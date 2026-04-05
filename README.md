# Monocular metric depth estimation (INR + depth prior)

Исследования: метрическая оценка глубины по одному изображению с использованием низкокачественного/частичного depth‑prior (ToF и др.) и локальной аффинной калибровки через implicit neural representations (INR).

## Быстрый старт

### 1. Окружение (uv)

Установите [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`

```bash
cd /Users/pevcov.artem2/Научка/depth_estimation
uv sync
```

Создаётся виртуальное окружение `.venv` и ставятся зависимости из `pyproject.toml`. Дальше запускайте скрипты через `uv run` (активация не обязательна):

```bash
uv run python scripts/run_depth_anything.py path/to/image.png
```

Или активируйте окружение вручную:

```bash
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python scripts/run_depth_anything.py path/to/image.png
```

### 2. Depth Anything — инференс локально

**Сервер не нужен.** Модель скачивается с Hugging Face при первом запуске и выполняется на вашей машине (CPU / CUDA / Apple Silicon MPS).

```bash
# Одно изображение
python scripts/run_depth_anything.py path/to/image.png --out-dir ./outputs

# Папка с изображениями
python scripts/run_depth_anything.py path/to/images/ --out-dir ./outputs --save-npy
```

Варианты модели: `--model LiheYoung/depth-anything-small-hf` (по умолчанию), `base`, `large`.  
На Apple Silicon используйте `float32` (скрипт уже переключает устройство на MPS автоматически).

### 3. Датасеты

#### NYU Depth V2 (indoor, метрическая глубина)

**Рекомендуется: официальный .mat** (Hugging Face больше не поддерживает dataset scripts для этого датасета.)

```bash
# Скачать .mat (~2.8 GB) и извлечь все кадры в data/nyu_depth_v2/labeled/
uv run python scripts/download_nyu.py --save-dir ./data

# Ограничить число сэмплов (например, 100 для теста):
uv run python scripts/download_nyu.py --save-dir ./data --max-samples 100
```

Если автоматическая загрузка не сработает, скачайте вручную [nyu_depth_v2_labeled.mat](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) и укажите путь:

```bash
uv run python scripts/download_nyu.py --save-dir ./data --mat-path /path/to/nyu_depth_v2_labeled.mat --max-samples 100
```

Результат: `data/nyu_depth_v2/labeled/images/` и `data/nyu_depth_v2/labeled/depth/` с PNG.

#### KITTI Depth / KITTI Raw

Нужна **регистрация** на [KITTI](https://www.cvlibs.net/datasets/kitti/). После этого скачайте данные с:

- [Depth completion benchmark](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion): devkit, annotated depth (~14 GB), velodyne (~5 GB), selection val/test (~2 GB).
- [Raw data](https://www.cvlibs.net/datasets/kitti/raw_data.php) — по датам/драйвам.

Создать структуру каталогов и посмотреть ссылки:

```bash
python scripts/download_kitti.py --save-dir ./data --info-only
python scripts/download_kitti.py --save-dir ./data --setup-dirs
```

После скачивания распакуйте архивы в `data/kitti_depth/` согласно описанию на сайте KITTI.

## Эксперименты на NYU (сравнение калибровок)

Единый раннер и JSON-конфиги: один прогон = несколько методов, общий инференс Depth Anything на кадр, метрики в `results.json`.  
Старые отдельные скрипты `eval_depth_anything_nyu.py` / `eval_local_calibration_nyu.py` удалены — используйте конфиги ниже.

**Только глобальная калибровка (20 кадров):**

```bash
uv run python -m scripts.run_experiment --config configs/exp_global_nyu.json
```

**Global + local (bilateral), как раньше в парном eval:**

```bash
uv run python -m scripts.run_experiment --config configs/exp_global_local_nyu.json
```

**Сравнение локального метода: без сглаживания / Гаусс / билатеральное (20 кадров):**

```bash
uv run python -m scripts.run_experiment --config configs/exp_local_smoothing_nyu20.json
```

**Сравнение итогов нескольких экспериментов:**

```bash
uv run python scripts/compare_experiments.py outputs/experiments/local_smoothing_nyu20/results.json
```

Подробности и параметры билатерального сглаживания полей `(s, t)` — в [configs/README.md](configs/README.md).

**Сетка по числу суперпикселей для `local_bilateral` (NYU, 20 кадров):**

```bash
uv run python -m scripts.run_experiment --config configs/exp_bilateral_segments_grid.json
```

Сетка **400–700** суперпикселей: `configs/exp_bilateral_segments_400_700.json`.  
Та же сетка 50–400 **без сглаживания** (local none): `configs/exp_local_none_segments_grid.json`.

**INR (простой MLP и FiLM по регионам)** — `configs/exp_inr_compare.json` (обучение на GPU/MPS на sparse prior, см. `configs/README.md`).

## Структура проекта

```
depth_estimation/                 # репозиторий
├── README.md
├── pyproject.toml
├── uv.lock
├── requirements.txt
├── .gitignore
├── depth_estimation/             # Python-пакет
│   ├── calibration/              # global / local / INR, реестр методов
│   ├── data/                     # NYU utils, симуляция prior
│   ├── evaluation/             # метрики, ExperimentRunner, визуализация
│   └── models/                   # Depth Anything (HF) inference
├── configs/                      # JSON экспериментов (+ README)
├── data/                         # датасеты (не в git, см. .gitignore)
├── outputs/                      # артефакты (не в git)
└── scripts/
    ├── run_experiment.py         # главная точка входа экспериментов
    ├── compare_experiments.py    # сравнение results.json
    ├── run_depth_anything.py     # инференс DA на фото/папку
    ├── download_nyu.py
    └── download_kitti.py
```

Импорты в коде: `depth_estimation.calibration.methods`, `depth_estimation.evaluation.experiment`, `depth_estimation.data.nyu_utils`, `depth_estimation.models.da_inference`.

## Ответ на вопрос «локально или сервер?»

**Depth Anything можно полностью запускать локально:**

- Модели с Hugging Face (`LiheYoung/depth-anything-*-hf`) при первом запуске скачиваются в кэш (например `~/.cache/huggingface/`).
- Инференс идёт на CPU, GPU (CUDA) или Apple Silicon (MPS) без внешнего API.
- Для больших моделей (large) желателен GPU или мощный Mac с MPS; small/base приемлемо работают и на CPU для единичных кадров.

Сервер нужен только если вы специально захотите поднять API (например, для веб-демо); для исследований и пайплайнов достаточно локального запуска скриптов выше.

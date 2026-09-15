# config.py
from pathlib import Path

# Пути к директориям
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR  # или BASE_DIR / "data", если файлы лежат в подпапке
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Создаем папки при необходимости
RESULTS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Параметры задачи
TARGET_COL = "unit_sales"
ID_COL = "id"
DATE_COL = "date"
SERIES_ID_COLS = ["store_nbr", "item_nbr"]

# Горизонт прогнозирования (строго по Kaggle)
FORECAST_HORIZON = 16

# Срез данных по времени для экономии RAM и ускорения
# Kaggle Test: 2017-08-16 -> 2017-08-31 (16 дней)
TRAIN_START_DATE = "2017-01-01"

# Протокол Out-Of-Time валидации (последние 16 дней доступного train)
VAL_START_DATE = "2017-07-26"
VAL_END_DATE = "2017-08-10"

# Веса для NWRMSLE (из описания соревнования)
PERISHABLE_WEIGHT = 1.25
NON_PERISHABLE_WEIGHT = 1.0

# Сид для воспроизводимости
RANDOM_SEED = 42
# config.py
from pathlib import Path

# Пути к директориям
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR  
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

RESULTS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)

TARGET_COL = "unit_sales"
ID_COL = "id"
DATE_COL = "date"
SERIES_ID_COLS = ["store_nbr", "item_nbr"]

# Горизонт прогнозирования 
FORECAST_HORIZON = 16

TRAIN_START_DATE = "2017-01-01"

#  Out-Of-Time валидация 
VAL_START_DATE = "2017-07-26"
VAL_END_DATE = "2017-08-10"

# Веса для NWRMSLE
PERISHABLE_WEIGHT = 1.25
NON_PERISHABLE_WEIGHT = 1.0

# Сид для воспроизводимости
RANDOM_SEED = 42
# src/data.py
import gc
from typing import Dict, Tuple
import numpy as np
import pandas as pd

import config


def get_dtypes() -> dict:
    return {
        "id": "uint32",
        "store_nbr": "uint8",
        "item_nbr": "uint32",
        "unit_sales": "float32",
        "onpromotion": "object",  
    }


def load_train_data(
    start_date: str = config.TRAIN_START_DATE,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    
    print(f"Загрузка train.csv (фильтрация: дата >= {start_date})...")
    train_path = config.DATA_DIR / "train.csv"

    chunks = []
    dtypes = get_dtypes()

    for chunk in pd.read_csv(
        train_path,
        dtype=dtypes,
        parse_dates=[config.DATE_COL],
        chunksize=chunksize,
    ):

        filtered_chunk = chunk[chunk[config.DATE_COL] >= start_date].copy()
        if not filtered_chunk.empty:
            filtered_chunk["onpromotion"] = (
                filtered_chunk["onpromotion"]
                .map({True: 1, "True": 1, 1: 1, False: 0, "False": 0, 0: 0})
                .fillna(0)
                .astype("int8")
            )

            filtered_chunk[config.TARGET_COL] = (
                filtered_chunk[config.TARGET_COL].clip(lower=0).astype("float32")
            )
            chunks.append(filtered_chunk)

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(
        f"Train успешно загружен: {len(df):,} строк. Использование RAM: {df.memory_usage().sum() / 1024**2:.1f} MB"
    )
    return df


def load_test_data() -> pd.DataFrame:

    print("Загрузка test.csv...")
    dtypes = {
        "id": "uint32",
        "store_nbr": "uint8",
        "item_nbr": "uint32",
        "onpromotion": "object",  
    }
    test_df = pd.read_csv(
        config.DATA_DIR / "test.csv",
        dtype=dtypes,
        parse_dates=[config.DATE_COL],
    )
    test_df["onpromotion"] = (
        test_df["onpromotion"]
        .map({True: 1, "True": 1, 1: 1, False: 0, "False": 0, 0: 0})
        .fillna(0)
        .astype("int8")
    )
    return test_df


def load_exogenous_data() -> Dict[str, pd.DataFrame]:

    print("Загрузка экзогенных файлов...")

    items = pd.read_csv(
        config.DATA_DIR / "items.csv",
        dtype={
            "item_nbr": "uint32",
            "family": "category",
            "class": "uint16",
            "perishable": "uint8",
        },
    )
    items["weight"] = np.where(
        items["perishable"] == 1,
        config.PERISHABLE_WEIGHT,
        config.NON_PERISHABLE_WEIGHT,
    ).astype("float32")

    # 2. Stores (магазины)
    stores = pd.read_csv(
        config.DATA_DIR / "stores.csv",
        dtype={
            "store_nbr": "uint8",
            "city": "category",
            "state": "category",
            "type": "category",
            "cluster": "uint8",
        },
    )

    # 3. Oil 
    oil = pd.read_csv(config.DATA_DIR / "oil.csv", parse_dates=["date"])
    full_date_range = pd.date_range(
        start=oil["date"].min(), end="2017-08-31", freq="D"
    )
    oil = (
        oil.set_index("date")
        .reindex(full_date_range)
        .rename_axis("date")
        .reset_index()
    )
    oil["dcoilwtico"] = (
        oil["dcoilwtico"].interpolate(method="linear").bfill().ffill()
    )

    # 4. Holidays 
    holidays = pd.read_csv(
        config.DATA_DIR / "holidays_events.csv", parse_dates=["date"]
    )
    holidays = holidays[holidays["transferred"] == False]
    holidays_unique = (
        holidays.drop_duplicates(subset=["date"])
        .assign(is_holiday=1)[["date", "is_holiday"]]
        .copy()
    )

    # 5. Transactions 
    transactions = pd.read_csv(
        config.DATA_DIR / "transactions.csv",
        parse_dates=["date"],
        dtype={"store_nbr": "uint8", "transactions": "uint16"},
    )

    return {
        "items": items,
        "stores": stores,
        "oil": oil,
        "holidays": holidays_unique,
        "transactions": transactions,
    }


def get_train_val_split(
    df: pd.DataFrame, items_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Разделяет данные на train и validation"""
    val_mask = (df[config.DATE_COL] >= config.VAL_START_DATE) & (
        df[config.DATE_COL] <= config.VAL_END_DATE
    )
    train_mask = df[config.DATE_COL] < config.VAL_START_DATE

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    val_df = val_df.merge(
        items_df[["item_nbr", "weight"]], on="item_nbr", how="left"
    )
    val_weights = (
        val_df["weight"].fillna(config.NON_PERISHABLE_WEIGHT).values
    )
    val_df.drop(columns=["weight"], inplace=True)

    print(f"Train split: {len(train_df):,} строк (до {config.VAL_START_DATE})")
    print(
        f"Validation split: {len(val_df):,} строк ({config.VAL_START_DATE} -> {config.VAL_END_DATE})"
    )

    return train_df, val_df, val_weights
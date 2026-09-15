# src/features.py
import gc
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

import config


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерация календарных признаков."""
    df["dayofweek"] = df[config.DATE_COL].dt.dayofweek.astype("int8")
    df["day"] = df[config.DATE_COL].dt.day.astype("int8")
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype("int8")
    return df


def merge_exogenous_features(
    df: pd.DataFrame, exo: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Объединение с экзогенными таблицами (товары, магазины, нефть, праздники)."""
    # 1. Товары
    df = df.merge(
        exo["items"][["item_nbr", "family", "class", "perishable"]],
        on="item_nbr",
        how="left",
    )

    # 2. Магазины
    df = df.merge(
        exo["stores"][["store_nbr", "type", "cluster"]],
        on="store_nbr",
        how="left",
    )

    # 3. Нефть
    df = df.merge(
        exo["oil"][["date", "dcoilwtico"]], on=config.DATE_COL, how="left"
    )
    df["dcoilwtico"] = df["dcoilwtico"].astype("float32")

    # 4. Праздники
    df = df.merge(
        exo["holidays"][["date", "is_holiday"]], on=config.DATE_COL, how="left"
    )
    df["is_holiday"] = df["is_holiday"].fillna(0).astype("int8")

    return df


def generate_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Генерация лагов и скользящих средних со сдвигом >= 16 дней."""
    print("Генерация лагов (16, 21, 28 дней и rolling mean 7)...")

    # Убираем возможные дубликаты пар дата-магазин-товар
    df = df.drop_duplicates(subset=[config.DATE_COL, "store_nbr", "item_nbr"])

    # Сводная таблица (дата x ряд) 
    pivot = df.pivot(
        index=config.DATE_COL,
        columns=["store_nbr", "item_nbr"],
        values=config.TARGET_COL,
    )

    lags = [16, 21, 28]
    lag_dfs = []

    for lag in lags:
        shifted = pivot.shift(lag).stack(["store_nbr", "item_nbr"]).reset_index()
        shifted.columns = [
            config.DATE_COL,
            "store_nbr",
            "item_nbr",
            f"sales_lag_{lag}",
        ]
        lag_dfs.append(shifted)

    # Скользящее среднее за 7 дней (сдвиг 16)
    roll_7 = (
        pivot.shift(16)
        .rolling(window=7, min_periods=1)
        .mean()
        .stack(["store_nbr", "item_nbr"])
        .reset_index()
    )
    roll_7.columns = [
        config.DATE_COL,
        "store_nbr",
        "item_nbr",
        "sales_roll_mean_7",
    ]
    lag_dfs.append(roll_7)

    # Соединяем лаги обратно в общий датасет
    for lag_df in lag_dfs:
        df = df.merge(
            lag_df, on=[config.DATE_COL, "store_nbr", "item_nbr"], how="left"
        )

    lag_cols = [f"sales_lag_{lag}" for lag in lags] + ["sales_roll_mean_7"]
    df[lag_cols] = df[lag_cols].fillna(0).astype("float32")

    del pivot, lag_dfs
    gc.collect()
    return df


def prepare_dataset(
    df: pd.DataFrame, exo: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, List[str]]:
    """Полный пайплайн сборки датасета для ML."""
    print("Формирование признаков...")
    df = add_calendar_features(df)
    df = merge_exogenous_features(df, exo)
    df = generate_lag_features(df)

    # Категориальные признаки приводим к типу category для LightGBM
    cat_features = ["store_nbr", "item_nbr", "family", "type", "cluster"]
    for col in cat_features:
        df[col] = df[col].astype("category")

    feature_cols = [
        "store_nbr",
        "item_nbr",
        "onpromotion",
        "dayofweek",
        "day",
        "is_weekend",
        "family",
        "class",
        "perishable",
        "type",
        "cluster",
        "dcoilwtico",
        "is_holiday",
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_roll_mean_7",
    ]

    return df, feature_cols
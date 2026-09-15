# make_naive_submission.py
import numpy as np
import pandas as pd
import config
from src.data import load_test_data


def make_naive_submission():

    # Загружаем только последнюю неделю трейна (с 2017-08-08 по 2017-08-15)
    print("Чтение последней недели продаж перед тестом...")
    dtypes = {
        "id": "uint32",
        "store_nbr": "uint8",
        "item_nbr": "uint32",
        "unit_sales": "float32",
    }

    chunks = []
    for chunk in pd.read_csv(
        config.DATA_DIR / "train.csv",
        dtype=dtypes,
        parse_dates=[config.DATE_COL],
        chunksize=1_000_000,
    ):
        filt = chunk[chunk[config.DATE_COL] >= "2017-08-08"].copy()
        if not filt.empty:
            filt[config.TARGET_COL] = filt[config.TARGET_COL].clip(lower=0)
            chunks.append(filt)

    recent_train = pd.concat(chunks, ignore_index=True)
    recent_train["dayofweek"] = recent_train[config.DATE_COL].dt.dayofweek

    # Считаем продажи по парам (магазин, товар, день недели)
    print("Построение карты продаж по дням недели")
    lookup = (
        recent_train.groupby(["store_nbr", "item_nbr", "dayofweek"])[
            config.TARGET_COL
        ]
        .last()
        .reset_index()
    )

    #  Загружаем test.csv
    test_df = load_test_data()
    test_df["dayofweek"] = test_df[config.DATE_COL].dt.dayofweek

    # Подтягиваем продажи соответствующего дня недели
    print("Формирование прогнозов")
    merged = test_df.merge(
        lookup, on=["store_nbr", "item_nbr", "dayofweek"], how="left"
    )

    # Если товара в этот день не было в магазине — прогнозируем 0
    merged[config.TARGET_COL] = merged[config.TARGET_COL].fillna(0)

    # Формируем сабмит
    sub = merged[[config.ID_COL, config.TARGET_COL]].copy()

    # Сохраняем файл
    sub_path = config.BASE_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    print(f"\n Файл успешно сохранен: {sub_path}")
    print(f"Всего строк: {len(sub):,}")
    print("\nПервые 10 строк вашего сабмита:")
    print(sub.head(10))


if __name__ == "__main__":
    make_naive_submission()
# make_smart_lgbm_submission.py
import gc
import sys

sys.modules["dask"] = None
sys.modules["dask.dataframe"] = None

import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from src.data import load_exogenous_data, load_test_data


def run_smart_pipeline():
    print("==================================================")
    print("   SMART LIGHTGBM: РЕШЕНИЕ ПРОБЛЕМЫ НУЛЕВЫХ ПРОДАЖ")
    print("==================================================")

    exo = load_exogenous_data()

    # 1. Читаем продажи за последние 60 дней перед тестом (с 2017-06-15)
    print("1. Загрузка последних 60 дней train.csv...")
    dtypes = {
        "id": "uint32",
        "store_nbr": "uint8",
        "item_nbr": "uint32",
        "unit_sales": "float32",
        "onpromotion": "object",
    }
    chunks = []
    for chunk in pd.read_csv(
        config.DATA_DIR / "train.csv",
        dtype=dtypes,
        parse_dates=[config.DATE_COL],
        chunksize=1_000_000,
    ):
        f = chunk[chunk[config.DATE_COL] >= "2017-06-15"].copy()
        if not f.empty:
            f[config.TARGET_COL] = f[config.TARGET_COL].clip(lower=0)
            f["onpromotion"] = (
                f["onpromotion"].map({True: 1, 1: 1}).fillna(0).astype("int8")
            )
            chunks.append(f)

    train_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    # 2. Создаем сводную таблицу (Store-Item x Date)
    print("2. Создание плотной матрицы продаж (дата x магазин-товар)...")
    # Добавляем логарифм продаж
    train_df["log_sales"] = np.log1p(train_df[config.TARGET_COL])

    pivot = train_df.pivot(
        index=["store_nbr", "item_nbr"],
        columns=config.DATE_COL,
        values="log_sales",
    ).fillna(0)

    # 3. Генерация мощных оконных признаков для каждого товара в магазине
    print("3. Расчет скользящих окон (3, 7, 14, 30, 60 дней)...")
    # Считаем статистики по окнам до 2017-08-15
    dates = pivot.columns

    X_train_list = []
    y_train_list = []

    # Создаем обучающую точку (например, срез на 2017-07-26, чтобы смоделировать прогноз на 16 дней)
    # И тестовую точку (срез на 2017-08-15)

    def extract_features(cutoff_date, p):
        # Окна до даты cutoff_date
        past_dates = [d for d in p.columns if d <= pd.to_datetime(cutoff_date)]
        p_past = p[past_dates]

        feat = pd.DataFrame(index=p.index)
        feat["mean_3"] = p_past.iloc[:, -3:].mean(axis=1)
        feat["mean_7"] = p_past.iloc[:, -7:].mean(axis=1)
        feat["mean_14"] = p_past.iloc[:, -14:].mean(axis=1)
        feat["mean_30"] = p_past.iloc[:, -30:].mean(axis=1)
        feat["mean_60"] = p_past.iloc[:, -60:].mean(axis=1)
        feat["zeros_14"] = (p_past.iloc[:, -14:] == 0).sum(axis=1)
        feat["zeros_30"] = (p_past.iloc[:, -30:] == 0).sum(axis=1)
        feat["last_val"] = p_past.iloc[:, -1]
        return feat

    # Признаки для обучения (по состоянию на 2017-07-25, таргет - следующие 16 дней)
    print("Формирование матриц обучения...")
    X_tr = extract_features("2017-07-25", pivot)
    # Таргет для трейна: средний лог продаж за 2017-07-26 .. 2017-08-10
    val_dates = [
        d
        for d in dates
        if d >= pd.to_datetime("2017-07-26")
        and d <= pd.to_datetime("2017-08-10")
    ]
    y_tr = pivot[val_dates].mean(axis=1).values

    # Признаки для инференса на Kaggle test (по состоянию на 2017-08-15)
    print("Формирование матриц для теста Kaggle...")
    X_te = extract_features("2017-08-15", pivot)

    # 4. Добавляем метаданные (товары, магазины)
    items = exo["items"].set_index("item_nbr")
    stores = exo["stores"].set_index("store_nbr")

    for df in [X_tr, X_te]:
        df["family"] = (
            df.index.get_level_values("item_nbr")
            .map(items["family"])
            .astype("category")
        )
        df["perishable"] = (
            df.index.get_level_values("item_nbr")
            .map(items["perishable"])
            .astype("int8")
        )
        df["class"] = (
            df.index.get_level_values("item_nbr")
            .map(items["class"])
            .astype("int16")
        )
        df["store_type"] = (
            df.index.get_level_values("store_nbr")
            .map(stores["type"])
            .astype("category")
        )
        df["cluster"] = (
            df.index.get_level_values("store_nbr")
            .map(stores["cluster"])
            .astype("int8")
        )

    # 5. Обучение LightGBM
    print("\n4. Обучение LightGBM на очищенных оконных фичах...")
    weights = np.where(X_tr["perishable"] == 1, 1.25, 1.0)

    model = lgb.LGBMRegressor(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=weights)

    # 6. Предсказание
    print("5. Генерация предсказаний...")
    pred_log = model.predict(X_te)
    pred_sales = np.expm1(np.clip(pred_log, 0, None))

    # СМАРТ-МАСКИРОВАНИЕ (ZERO-MASKING)
    # Если за последние 30 дней товар не продался ни разу -> строго 0!
    inactive_mask = X_te["mean_30"].values == 0
    pred_sales[inactive_mask] = 0.0

    # Зануляем околонулевой шум
    pred_sales[pred_sales < 0.08] = 0.0

    # Создаем итоговый lookup для предсказаний
    pred_df = pd.DataFrame(
        {"unit_sales": pred_sales}, index=X_te.index
    ).reset_index()

    # 7. Загрузка test.csv и матчинг
    print("6. Формирование submission.csv...")
    test_df = load_test_data()
    test_df["dayofweek"] = test_df[config.DATE_COL].dt.dayofweek

    # Мерджим предсказания базового уровня продаж
    merged = test_df.merge(
        pred_df, on=["store_nbr", "item_nbr"], how="left"
    ).fillna(0)

    # Учитываем эффект промо-акций (товары на промо продаются в среднем на 25% лучше)
    merged.loc[merged["onpromotion"] == 1, "unit_sales"] *= 1.25

    # Сохраняем сабмит
    sub = merged[[config.ID_COL, "unit_sales"]]
    sub_path = config.BASE_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    print("\n==================================================")
    print(f" САБМИТ УСПЕШНО ОБНОВЛЕН: {sub_path}")
    print(f"Всего строк: {len(sub):,}")
    print("==================================================")
    print(sub.head(10))


if __name__ == "__main__":
    run_smart_pipeline()
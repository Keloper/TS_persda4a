# make_improved_lgbm_submission.py
import gc
import sys

sys.modules["dask"] = None
sys.modules["dask.dataframe"] = None

import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from src.data import load_exogenous_data, load_test_data


def run_improved_lgbm():

    exo = load_exogenous_data()

    print(" Загрузка train.csv")
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
        f = chunk[chunk[config.DATE_COL] >= "2017-05-01"].copy()
        if not f.empty:
            f[config.TARGET_COL] = f[config.TARGET_COL].clip(lower=0)
            f["onpromotion"] = (
                f["onpromotion"].map({True: 1, 1: 1}).fillna(0).astype("int8")
            )
            chunks.append(f)

    train_df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    test_df = load_test_data()

    train_df["log_sales"] = np.log1p(train_df[config.TARGET_COL])

    # Матрица продаж
    sales_pivot = train_df.pivot(
        index=["store_nbr", "item_nbr"],
        columns=config.DATE_COL,
        values="log_sales",
    ).fillna(0)

    # Матрица промо-акций
    promo_all = pd.concat(
        [
            train_df[["store_nbr", "item_nbr", config.DATE_COL, "onpromotion"]],
            test_df[["store_nbr", "item_nbr", config.DATE_COL, "onpromotion"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["store_nbr", "item_nbr", config.DATE_COL])

    promo_pivot = promo_all.pivot(
        index=["store_nbr", "item_nbr"],
        columns=config.DATE_COL,
        values="onpromotion",
    ).fillna(0)

    del train_df, promo_all
    gc.collect()

    items = exo["items"].set_index("item_nbr")
    stores = exo["stores"].set_index("store_nbr")

    # 3. Функция генерации признаков для среза дат
    def create_features_for_cutoff(cutoff_date, is_train=True):
        cutoff = pd.to_datetime(cutoff_date)
        past_dates = [d for d in sales_pivot.columns if d <= cutoff]

        p_sales = sales_pivot[past_dates]

        feat = pd.DataFrame(index=sales_pivot.index)

        # Скользящие средние продаж
        feat["mean_3"] = p_sales.iloc[:, -3:].mean(axis=1)
        feat["mean_7"] = p_sales.iloc[:, -7:].mean(axis=1)
        feat["mean_14"] = p_sales.iloc[:, -14:].mean(axis=1)
        feat["mean_30"] = p_sales.iloc[:, -30:].mean(axis=1)
        feat["mean_60"] = p_sales.iloc[:, -60:].mean(axis=1)

        # Выходные против будней за последние 14 дней
        last_14_dates = past_dates[-14:]
        weekend_dates = [d for d in last_14_dates if d.dayofweek in [5, 6]]
        weekday_dates = [d for d in last_14_dates if d.dayofweek not in [5, 6]]
        feat["mean_weekend"] = p_sales[weekend_dates].mean(axis=1)
        feat["mean_weekday"] = p_sales[weekday_dates].mean(axis=1)

        # Доля нулей (индикатор затухания спроса)
        feat["zeros_14"] = (p_sales.iloc[:, -14:] == 0).sum(axis=1)
        feat["zeros_30"] = (p_sales.iloc[:, -30:] == 0).sum(axis=1)
        feat["has_recent_sales"] = (feat["mean_7"] > 0).astype("int8")

        # Промо-акции в прошлом (за 14 дней)
        past_promo_dates = [d for d in promo_pivot.columns if d in past_dates[-14:]]
        feat["promo_history_14"] = promo_pivot[past_promo_dates].sum(axis=1)

        # Будущие промо-акции (в окне следующих 16 дней!)
        future_dates = [
            d
            for d in promo_pivot.columns
            if (d > cutoff) and (d <= cutoff + pd.Timedelta(days=16))
        ]
        feat["promo_forward_16"] = promo_pivot[future_dates].sum(axis=1)

        # Метаданные товаров и магазинов
        feat["family"] = (
            feat.index.get_level_values("item_nbr")
            .map(items["family"])
            .astype("category")
        )
        feat["perishable"] = (
            feat.index.get_level_values("item_nbr")
            .map(items["perishable"])
            .astype("int8")
        )
        feat["store_type"] = (
            feat.index.get_level_values("store_nbr")
            .map(stores["type"])
            .astype("category")
        )
        feat["cluster"] = (
            feat.index.get_level_values("store_nbr")
            .map(stores["cluster"])
            .astype("int8")
        )

        if is_train:
            # Таргет: средний лог продаж за следующие 16 дней
            target_dates = [
                d
                for d in sales_pivot.columns
                if (d > cutoff) and (d <= cutoff + pd.Timedelta(days=16))
            ]
            y = sales_pivot[target_dates].mean(axis=1).values
            return feat, y
        return feat

    # 4. Формирование обучающей выборки на 3 срезах дат
    print(" Извлечение признаков на 3 исторических срезах...")
    cutoffs = ["2017-06-21", "2017-07-05", "2017-07-25"]
    X_trains, y_trains = [], []

    for c in cutoffs:
        print(f"   Обработка среза: {c}")
        X_c, y_c = create_features_for_cutoff(c, is_train=True)
        X_trains.append(X_c)
        y_trains.append(y_c)

    X_train = pd.concat(X_trains, axis=0)
    y_train = np.concatenate(y_trains)

    # Признаки для тестового сабмита на Kaggle 
    print("   Извлечение признаков для теста (срез: 2017-08-15)...")
    X_test = create_features_for_cutoff("2017-08-15", is_train=False)

    # Веса для скоропортящихся товаров
    weights = np.where(X_train["perishable"] == 1, 1.25, 1.0)

    # Обучение модели
    print(
        f"\n Обучение  LightGBM "
    )
    model = lgb.LGBMRegressor(
        n_estimators=450,
        learning_rate=0.04,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=weights)

    # Инференс и пост-процессинг
    pred_log = model.predict(X_test)
    pred_sales = np.expm1(np.clip(pred_log, 0, None))

    # Если товар не продавался в магазине последние 60 дней ->  0
    inactive = X_test["mean_60"].values == 0
    pred_sales[inactive] = 0.0

    pred_sales[pred_sales < 0.05] = 0.0

    pred_df = pd.DataFrame(
        {"base_pred": pred_sales}, index=X_test.index
    ).reset_index()

    # Финальное объединение 
    sub_df = test_df.merge(
        pred_df, on=["store_nbr", "item_nbr"], how="left"
    ).fillna(0)

    # Учитываем индивидуальный факт акции в конкретный день
    sub_df[config.TARGET_COL] = sub_df["base_pred"]
    sub_df.loc[sub_df["onpromotion"] == 1, config.TARGET_COL] *= 1.30

    # Сохранение
    final_sub = sub_df[[config.ID_COL, config.TARGET_COL]]
    sub_path = config.BASE_DIR / "submission.csv"
    final_sub.to_csv(sub_path, index=False)

    print(f" сабмит сохранен: {sub_path}")
    print(final_sub.head(10))


if __name__ == "__main__":
    run_improved_lgbm()
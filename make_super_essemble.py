# make_killer_ensemble.py
import gc
import sys

sys.modules["dask"] = None
sys.modules["dask.dataframe"] = None

import lightgbm as lgb
import numpy as np
import pandas as pd
import config
from src.data import load_exogenous_data, load_test_data


def run_super_ensemble():
    print("==================================================")
    print("     KILLER ENSEMBLE: DOW-ПРОФИЛИ, ЭКВАДОРСКИЕ    ")
    print("        ЗАРПЛАТЫ И БЛЕНДИНГ С НЕЙРОСЕТЬЮ          ")
    print("==================================================")

    exo = load_exogenous_data()

    # 1. Загрузка train с 1 мая 2017
    print("1. Загрузка данных...")
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

    # 2. Расчет коэффициентов дней недели (DOW Profiles)
    print("2. Расчет сезонных профилей дней недели (DOW Profiles)...")
    train_df["dayofweek"] = train_df[config.DATE_COL].dt.dayofweek

    # Добавляем категорию товара
    items_map = exo["items"].set_index("item_nbr")["family"].to_dict()
    train_df["family"] = train_df["item_nbr"].map(items_map)

    # Средние продажи по (магазин, семья, день недели)
    dow_sales = (
        train_df.groupby(["store_nbr", "family", "dayofweek"])[
            config.TARGET_COL
        ]
        .mean()
        .reset_index()
    )
    # Средние продажи по (магазин, семья) за всю неделю
    week_sales = (
        train_df.groupby(["store_nbr", "family"])[config.TARGET_COL]
        .mean()
        .reset_index()
        .rename(columns={config.TARGET_COL: "mean_week_sales"})
    )

    dow_profiles = dow_sales.merge(
        week_sales, on=["store_nbr", "family"], how="left"
    )
    # Коэффициент отклонения дня недели (например, 1.3 в субботу, 0.8 в понедельник)
    dow_profiles["dow_coeff"] = (
        dow_profiles[config.TARGET_COL] / (dow_profiles["mean_week_sales"] + 1e-5)
    ).clip(0.4, 2.0)
    dow_map = (
        dow_profiles.set_index(["store_nbr", "family", "dayofweek"])[
            "dow_coeff"
        ]
        .to_dict()
    )

    # 3. Подготовка матриц для LightGBM
    print("3. Обучение базового бустинга...")
    train_df["log_sales"] = np.log1p(train_df[config.TARGET_COL])
    sales_pivot = train_df.pivot(
        index=["store_nbr", "item_nbr"],
        columns=config.DATE_COL,
        values="log_sales",
    ).fillna(0)

    # Обучаем модель на срезе 25 июля
    past_dates_tr = [
        d for d in sales_pivot.columns if d <= pd.to_datetime("2017-07-25")
    ]
    target_dates_tr = [
        d
        for d in sales_pivot.columns
        if d >= pd.to_datetime("2017-07-26")
        and d <= pd.to_datetime("2017-08-10")
    ]

    p_past_tr = sales_pivot[past_dates_tr]
    X_tr = pd.DataFrame(index=sales_pivot.index)
    X_tr["mean_3"] = p_past_tr.iloc[:, -3:].mean(axis=1)
    X_tr["mean_7"] = p_past_tr.iloc[:, -7:].mean(axis=1)
    X_tr["mean_14"] = p_past_tr.iloc[:, -14:].mean(axis=1)
    X_tr["mean_30"] = p_past_tr.iloc[:, -30:].mean(axis=1)
    X_tr["mean_60"] = p_past_tr.iloc[:, -60:].mean(axis=1)
    X_tr["zeros_14"] = (p_past_tr.iloc[:, -14:] == 0).sum(axis=1)
    X_tr["zeros_30"] = (p_past_tr.iloc[:, -30:] == 0).sum(axis=1)
    y_tr = sales_pivot[target_dates_tr].mean(axis=1).values

    # Тестовые фичи на 15 августа
    p_past_te = sales_pivot
    X_te = pd.DataFrame(index=sales_pivot.index)
    X_te["mean_3"] = p_past_te.iloc[:, -3:].mean(axis=1)
    X_te["mean_7"] = p_past_te.iloc[:, -7:].mean(axis=1)
    X_te["mean_14"] = p_past_te.iloc[:, -14:].mean(axis=1)
    X_te["mean_30"] = p_past_te.iloc[:, -30:].mean(axis=1)
    X_te["mean_60"] = p_past_te.iloc[:, -60:].mean(axis=1)
    X_te["zeros_14"] = (p_past_te.iloc[:, -14:] == 0).sum(axis=1)
    X_te["zeros_30"] = (p_past_te.iloc[:, -30:] == 0).sum(axis=1)

    items = exo["items"].set_index("item_nbr")
    for d in [X_tr, X_te]:
        d["family"] = (
            d.index.get_level_values("item_nbr")
            .map(items["family"])
            .astype("category")
        )
        d["perishable"] = (
            d.index.get_level_values("item_nbr")
            .map(items["perishable"])
            .astype("int8")
        )

    weights = np.where(X_tr["perishable"] == 1, 1.25, 1.0)
    model = lgb.LGBMRegressor(
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=weights)

    # 4. Базовый прогноз бустинга
    pred_base = np.expm1(np.clip(model.predict(X_te), 0, None))
    inactive_mask = X_te["mean_60"].values == 0
    pred_base[inactive_mask] = 0.0
    pred_base[pred_base < 0.05] = 0.0

    lgb_pred_df = pd.DataFrame(
        {"base_sales": pred_base}, index=X_te.index
    ).reset_index()

    # 5. Разворачиваем прогноз на дни с учетом DOW и Зарплаты
    print("4. Модуляция прогноза по дням недели и дням зарплаты...")
    test_df["dayofweek"] = test_df[config.DATE_COL].dt.dayofweek
    test_df["day"] = test_df[config.DATE_COL].dt.day
    test_df["family"] = test_df["item_nbr"].map(items_map)

    merged = test_df.merge(
        lgb_pred_df, on=["store_nbr", "item_nbr"], how="left"
    ).fillna(0)

    # Применяем коэффициент дня недели
    dow_factors = [
        dow_map.get((s, f, d), 1.0)
        for s, f, d in zip(
            merged["store_nbr"], merged["family"], merged["dayofweek"]
        )
    ]
    merged["lgb_pred"] = merged["base_sales"] * np.array(dow_factors)

    # Учитываем промо (+25%)
    merged.loc[merged["onpromotion"] == 1, "lgb_pred"] *= 1.25

    # ЭФФЕКТ ЭКВАДОРСКОЙ ЗАРПЛАТЫ (31 августа = Payday Spike +22%)
    payday_mask = merged["day"] == 31
    merged.loc[payday_mask, "lgb_pred"] *= 1.22
    print(
        f"   Применен повышающий коэффициент зарплаты к {payday_mask.sum():,} записям за 31 августа!"
    )

    # 6. БЛЕНДИНГ С НЕЙРОСЕТЬЮ (если есть submission_dl.csv)
    dl_path = config.PREDICTIONS_DIR / "submission_dl.csv"
    if dl_path.exists():
        print("5. Обнаружен сабмит нейросети! Запуск блендинга (85% LGB + 15% DL)...")
        dl_sub = pd.read_csv(dl_path)
        merged["dl_pred"] = dl_sub[config.TARGET_COL].values
        # Взвешенное среднее
        final_preds = 0.85 * merged["lgb_pred"] + 0.15 * merged["dl_pred"]
    else:
        print("5. Сабмит нейросети не найден, используем чистый улучшенный LightGBM.")
        final_preds = merged["lgb_pred"]

    # 7. Формирование сабмита
    final_preds = np.clip(final_preds, 0, None)
    sub = pd.DataFrame({config.ID_COL: test_df[config.ID_COL], config.TARGET_COL: final_preds})

    sub_path = config.BASE_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    print("\n==================================================")
    print(f" СУПЕР-АНСАМБЛЬ УСПЕШНО СОЗДАН: {sub_path}")
    print(f"Всего строк: {len(sub):,}")
    print("==================================================")
    print(sub.head(10))


if __name__ == "__main__":
    run_super_ensemble()
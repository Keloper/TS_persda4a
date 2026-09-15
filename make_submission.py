# make_submission.py
import gc
import sys

# Блокировка dask на случай конфликта в Anaconda
sys.modules["dask"] = None
sys.modules["dask.dataframe"] = None

import numpy as np
import pandas as pd
import config
from src.data import load_exogenous_data, load_test_data, load_train_data
from src.features import prepare_dataset
from src.models import MLForecaster


def generate_submission():
    print("==================================================")
    print("      ГЕНЕРАЦИЯ САБМИТА ДЛЯ KAGGLE (LightGBM)     ")
    print("==================================================")

    # 1. Загрузка экзогенных данных
    exo = load_exogenous_data()

    # 2. Загружаем Train (с 1 июня 2017 — оптимально по памяти и качеству)
    train_df = load_train_data(start_date="2017-06-01")

    # 3. Загружаем Test (16–31 августа 2017)
    test_df = load_test_data()
    test_df[config.TARGET_COL] = 0.0  # временный плейсхолдер для таргета

    test_ids = test_df[config.ID_COL].values

    # 4. Объединяем для сквозного расчета лагов и внешних фичей
    print("\nОбъединение Train и Test для генерации признаков...")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    del train_df, test_df
    gc.collect()

    # 5. Извлечение признаков (календарь, нефть, праздники, магазины, лаги)
    combined_df, feature_cols = prepare_dataset(combined_df, exo)

    # 6. Разделяем обратно на train и test строго по дате
    train_mask = combined_df[config.DATE_COL] <= "2017-08-15"
    test_mask = combined_df[config.DATE_COL] >= "2017-08-16"

    X_train = combined_df.loc[train_mask, feature_cols]
    y_train = combined_df.loc[train_mask, config.TARGET_COL].values

    X_test = combined_df.loc[test_mask, feature_cols]

    # Веса для скоропортящихся товаров (perishable = 1.25)
    train_items = combined_df.loc[train_mask, "item_nbr"]
    weights_series = train_items.map(
        exo["items"].set_index("item_nbr")["weight"]
    ).fillna(1.0)
    train_weights = weights_series.values

    del combined_df
    gc.collect()

    # 7. Обучение LightGBM
    print(f"\nОбучение LightGBM на полном Train ({len(X_train):,} строк)...")
    model = MLForecaster(n_estimators=300, learning_rate=0.08)
    model.fit(X_train, y_train, sample_weight=train_weights)

    # 8. Инференс на тесте
    print("\nГенерация предсказаний для test.csv...")
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # клипаем отрицательные значения до 0

    # 9. Сохранение файла сабмита
    sub = pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: preds})

    sub_path_root = config.BASE_DIR / "submission.csv"
    sub_path_res = config.PREDICTIONS_DIR / "submission_lightgbm.csv"

    sub.to_csv(sub_path_root, index=False)
    sub.to_csv(sub_path_res, index=False)

    print("\n==================================================")
    print(f" САБМИТ УСПЕШНО СОЗДАН: {sub_path_root}")
    print(f"Всего предсказаний: {len(sub):,} (ожидалось ровно 3,370,464)")
    print("==================================================")
    print("\nПервые 10 строк финального прогноза:")
    print(sub.head(10))


if __name__ == "__main__":
    generate_submission()
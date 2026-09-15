# run_stage4_ml.py
import config
from src.data import get_train_val_split, load_exogenous_data, load_train_data
from src.features import prepare_dataset
from src.metrics import evaluate_all_metrics
from src.models import MLForecaster

# 1. Загрузка данных и экзогенных таблиц
exo = load_exogenous_data()
train = load_train_data(start_date="2017-05-15")

# 2. Генерация признаков (лаги, скользящие средние, календарь, внешние фичи)
df_feat, feature_cols = prepare_dataset(train, exo)

# 3. Валидационный сплит (последние 16 дней)
train_df, val_df, val_weights = get_train_val_split(df_feat, exo["items"])

# Веса для скоропортящихся товаров (perishable)
train_df = train_df.merge(
    exo["items"][["item_nbr", "weight"]], on="item_nbr", how="left"
)
train_weights = train_df["weight"].fillna(1.0).values

X_train = train_df[feature_cols]
y_train = train_df[config.TARGET_COL].values

X_val = val_df[feature_cols]
y_val = val_df[config.TARGET_COL].values

# 4. Обучение модели LightGBM
print(f"\nОбучение LightGBM на {len(X_train):,} строках...")
model = MLForecaster(n_estimators=250, learning_rate=0.08)
model.fit(X_train, y_train, sample_weight=train_weights)

# 5. Прогноз и расчет метрик
print("\nРасчет предсказаний на валидационной выборке...")
val_preds = model.predict(X_val)

metrics_ml = evaluate_all_metrics(y_val, val_preds, val_weights)

# 6. Сравнение с бейзлайнами из Этапа 3
print("\n==================================================")
print("          РЕЗУЛЬТАТЫ ЭТАПА 4 (Сравнение)          ")
print("==================================================")
print(f"Naive Baseline:          NWRMSLE = 0.9198 | WAPE = 0.6509")
print(f"Seasonal Naive (7d):     NWRMSLE = 0.8723 | WAPE = 0.6092")
print(
    f"LightGBM (ML + фичи):    NWRMSLE = {metrics_ml['NWRMSLE']:.4f} | WAPE = {metrics_ml['WAPE']:.4f} | MAE = {metrics_ml['MAE']:.4f}"
)
print("==================================================")

# 7. Важность признаков (для отчета)
print("\nТоп-10 самых важных признаков (Feature Importance):")
fi = model.get_feature_importances()
print(fi.head(10).to_string(index=False))

# Сохраняем важность признаков
fi.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
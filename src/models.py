# src/models.py
import sys

sys.modules["dask"] = None
sys.modules["dask.dataframe"] = None
import lightgbm as lgb
import numpy as np
import pandas as pd

import config


class NaiveBaseline:

    def __init__(self):
        self.last_sales = None

    def fit(self, train_df: pd.DataFrame):
        last_date = train_df[config.DATE_COL].max()
        last_day_df = train_df[train_df[config.DATE_COL] == last_date]
        self.last_sales = (
            last_day_df.groupby(["store_nbr", "item_nbr"])[config.TARGET_COL]
            .last()
            .reset_index()
        )
        return self

    def predict(self, val_df: pd.DataFrame) -> np.ndarray:
        merged = val_df[["store_nbr", "item_nbr"]].merge(
            self.last_sales, on=["store_nbr", "item_nbr"], how="left"
        )
        return merged[config.TARGET_COL].fillna(0).values


class SeasonalNaiveBaseline:

    def __init__(self, season_length: int = 7):
        self.season_length = season_length
        self.history = None

    def fit(self, train_df: pd.DataFrame):
        last_date = train_df[config.DATE_COL].max()
        cutoff_date = last_date - pd.Timedelta(days=self.season_length - 1)
        recent_df = train_df[train_df[config.DATE_COL] >= cutoff_date].copy()
        recent_df["dayofweek"] = recent_df[config.DATE_COL].dt.dayofweek
        self.history = (
            recent_df.groupby(["store_nbr", "item_nbr", "dayofweek"])[
                config.TARGET_COL
            ]
            .mean()
            .reset_index()
        )
        return self

    def predict(self, val_df: pd.DataFrame) -> np.ndarray:
        val_temp = val_df[
            ["store_nbr", "item_nbr", config.DATE_COL]
        ].copy()
        val_temp["dayofweek"] = val_temp[config.DATE_COL].dt.dayofweek
        merged = val_temp.merge(
            self.history,
            on=["store_nbr", "item_nbr", "dayofweek"],
            how="left",
        )
        return merged[config.TARGET_COL].fillna(0).values


class MLForecaster:
    """Градиентный бустинг LightGBM """

    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.08):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        self.feature_cols = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        print(
            f"Обучение LightGBM на {len(X_train):,} строках с {X_train.shape[1]} признаками..."
        )
        y_log = np.log1p(np.clip(y_train, 0, None))
        self.model.fit(X_train, y_log, sample_weight=sample_weight)
        self.feature_cols = list(X_train.columns)
        return self

    def predict(self, X_val: pd.DataFrame) -> np.ndarray:
        pred_log = self.model.predict(X_val)
        return np.clip(np.expm1(pred_log), 0, None)

    def get_feature_importances(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "feature": self.feature_cols,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
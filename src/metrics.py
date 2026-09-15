# src/metrics.py
import numpy as np


def compute_nwrmsle(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    """Нормализованный взвешенный корень из среднеквадратичной логарифмической ошибки.

    NWRMSLE = sqrt( sum(w_i * (ln(y_hat + 1) - ln(y + 1))^2) / sum(w_i) )
    """
    # Защита от отрицательных значений прогноза
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)

    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)

    weighted_squared_error = weights * (log_pred - log_true) ** 2
    sum_weights = np.sum(weights)

    if sum_weights == 0:
        return 0.0

    return float(np.sqrt(np.sum(weighted_squared_error) / sum_weights))


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (WAPE) = sum(|y - y_hat|) / sum(y)

    Показывает суммарную ошибку в % от общего объема продаж.
    """
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    total_sales = np.sum(y_true)
    if total_sales == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_sales)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (в штуках товара)."""
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> dict:
    """Считает все метрики разом и возвращает словарь."""
    return {
        "NWRMSLE": compute_nwrmsle(y_true, y_pred, weights),
        "WAPE": compute_wape(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
    }
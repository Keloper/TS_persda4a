# Corporación Favorita Grocery Sales Forecasting (HW3)

Решение задачи многомерного прогнозирования временных рядов потребительского спроса сети супермаркетов **Corporación Favorita** (Kaggle) с использованием экзогенных признаков, градиентного бустинга и глубокого обучения.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Оглавление
- [Описание задачи](#-описание-задачи)
- [Структура репозитория](#-структура-репозитория)
---

## 📖 Описание задачи

* **Цель:** Прогнозирование продаж `unit_sales` по ~170 000 временным рядам (парам «магазин — товар») на горизонт **16 дней** (с 16 по 31 августа 2017 года).
* **Целевая метрика:** **NWRMSLE** (Normalized Weighted Root Mean Squared Logarithmic Error) с весами $w_i = 1.25$ для скоропортящихся продуктов (`perishable`) и $w_i = 1.0$ для остальных.
* **Дополнительные метрики:** WAPE (Weighted Absolute Percentage Error) и MAE в физических штуках.

---

## 📂 Структура репозитория

Структура проекта организована в строгом соответствии с требованиями к воспроизводимости:

```text
favorita_forecasting/
├── config.py                 # Централизованная конфигурация (пути, гиперпараметры, даты сплитов)
├── requirements.txt          # Зафиксированные версии библиотек
├── run_stage4_ml.py          # Эксперимент: обучение и валидация LightGBM
├── run_stage5_dl.py          # Эксперимент: обучение и валидация PyTorch Entity Embeddings
├── make_naive_submission.py  # Генерация наивного сабмита (Seasonal Naive 7d)
├── make_dl_submission.py     # Генерация сабмита нейросети (PyTorch)
├── src/
│   ├── __init__.py
│   ├── config.py             # Ссылка/импорт глобального конфига
│   ├── data.py               # Потоковая загрузка данных (chunks), оптимизация памяти, time-based split
│   ├── features.py           # Генерация лагов (>=16d), скользящих окон, слияние экзогенных таблиц
│   ├── metrics.py            # Векторизованный расчет NWRMSLE, WAPE, MAE с восстановлением масштаба
│   └── models.py             # Классы бейзлайнов (Naive, SNaive), LightGBM и PyTorch EmbeddingNet
└── results/
    ├── comparison_metrics.csv # Сводная таблица результатов валидации
    ├── feature_importance.csv # Таблица важности признаков
    ├── analysis_results.ipynb # Ноутбук для отрисовки графиков и визуализаций
    └── predictions/           # Директория с локальными предсказаниями моделей

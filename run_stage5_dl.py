# run_stage5_dl.py
import gc
import sys

sys.modules["dask"] = None
sys.modules["dask.dataframe"] = None

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import config
from src.data import load_exogenous_data, load_train_data, get_train_val_split
from src.features import prepare_dataset
from src.metrics import evaluate_all_metrics

print("==================================================")
print("     ЭТАП 5: ОБУЧЕНИЕ И ВАЛИДАЦИЯ DEEP LEARNING   ")
print("==================================================")


# 1. Dataset для PyTorch
class TabularTSDataset(Dataset):

    def __init__(self, cats, conts, targets=None, weights=None):
        self.cats = torch.tensor(cats, dtype=torch.long)
        self.conts = torch.tensor(conts, dtype=torch.float32)
        self.targets = (
            torch.tensor(targets, dtype=torch.float32)
            if targets is not None
            else None
        )
        self.weights = (
            torch.tensor(weights, dtype=torch.float32)
            if weights is not None
            else None
        )

    def __len__(self):
        return len(self.cats)

    def __getitem__(self, idx):
        if self.targets is not None:
            return (
                self.cats[idx],
                self.conts[idx],
                self.targets[idx],
                self.weights[idx],
            )
        return self.cats[idx], self.conts[idx]


# 2. Архитектура нейросети с Entity Embeddings
class TimeSeriesEmbeddingNet(nn.Module):

    def __init__(
        self,
        cat_dims: list,
        emb_dims: list,
        num_conts: int,
        hidden_dims=[128, 64],
    ):
        super().__init__()
        # Слой эмбеддингов для каждой категории
        self.embeddings = nn.ModuleList(
            [nn.Embedding(c, d) for c, d in zip(cat_dims, emb_dims)]
        )
        total_emb_dim = sum(emb_dims)

        layers = []
        in_dim = total_emb_dim + num_conts
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                ]
            )
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cats, conts):
        embs = [
            emb_layer(cats[:, i])
            for i, emb_layer in enumerate(self.embeddings)
        ]
        x = torch.cat(embs + [conts], dim=1)
        return self.mlp(x).squeeze(-1)


# 3. Основной пайплайн
def run_dl():
    exo = load_exogenous_data()
    train = load_train_data(start_date="2017-06-01")

    # Формируем признаки
    df_feat, feature_cols = prepare_dataset(train, exo)

    # Валидационный сплит
    train_df, val_df, val_weights = get_train_val_split(df_feat, exo["items"])

    train_df = train_df.merge(
        exo["items"][["item_nbr", "weight"]], on="item_nbr", how="left"
    )
    train_weights = train_df["weight"].fillna(1.0).values

    # Разделяем на категориальные и вещественные признаки
    cat_cols = ["store_nbr", "family", "type", "cluster"]
    cont_cols = [c for c in feature_cols if c not in cat_cols and c != "item_nbr"]

    print(
        f"\nКатегориальные фичи ({len(cat_cols)}): {cat_cols}"
    )
    print(
        f"Числовые/экзогенные фичи ({len(cont_cols)}): {cont_cols}"
    )

    # Нормализация категорий в диапазон 0..K-1
    cat_maps = {
        col: {val: i for i, val in enumerate(train_df[col].unique())}
        for col in cat_cols
    }
    cats_train = np.column_stack(
        [train_df[c].map(cat_maps[c]).fillna(0).values for c in cat_cols]
    )
    cats_val = np.column_stack(
        [val_df[c].map(cat_maps[c]).fillna(0).values for c in cat_cols]
    )

    # Нормализация числовых признаков (Mean-Std scaling)
    means = train_df[cont_cols].mean()
    stds = train_df[cont_cols].std().replace(0, 1)

    conts_train = ((train_df[cont_cols] - means) / stds).values.astype(
        np.float32
    )
    conts_val = ((val_df[cont_cols] - means) / stds).values.astype(np.float32)

    # Логарифм таргета
    y_train_log = np.log1p(train_df[config.TARGET_COL].values)
    y_val_real = val_df[config.TARGET_COL].values

    # PyTorch DataLoaders
    batch_size = 4096
    train_ds = TabularTSDataset(
        cats_train, conts_train, y_train_log, train_weights
    )
    val_ds = TabularTSDataset(cats_val, conts_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # Определение устройства (Apple MPS / CUDA / CPU)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Запуск обучения на устройстве: {device}")

    # Размеры эмбеддингов
    cat_dims = [len(cat_maps[c]) + 1 for c in cat_cols]
    emb_dims = [min(16, max(4, (c + 1) // 2)) for c in cat_dims]

    model = TimeSeriesEmbeddingNet(
        cat_dims, emb_dims, len(cont_cols), hidden_dims=[128, 64]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")

    # 4. Цикл обучения (3 эпохи для быстрого схождения)
    epochs = 3
    model.train()
    print("\nСтарт эпох обучения...")
    for epoch in range(epochs):
        total_loss = 0.0
        for b_cats, b_conts, b_y, b_w in train_loader:
            b_cats, b_conts = b_cats.to(device), b_conts.to(device)
            b_y, b_w = b_y.to(device), b_w.to(device)

            optimizer.zero_grad()
            preds = model(b_cats, b_conts)
            loss = torch.mean(b_w * criterion(preds, b_y))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(
            f"Эпоха {epoch+1}/{epochs} - Средний Loss (Weighted MSE): {total_loss / len(train_loader):.4f}"
        )

    # 5. Валидационный инференс
    print("\nИнференс DL модели на валидации...")
    model.eval()
    preds_list = []
    with torch.no_grad():
        for b_cats, b_conts in val_loader:
            b_cats, b_conts = b_cats.to(device), b_conts.to(device)
            preds = model(b_cats, b_conts)
            preds_list.append(preds.cpu().numpy())

    pred_log = np.concatenate(preds_list)
    val_preds = np.clip(np.expm1(pred_log), 0, None)

    # Расчет метрик
    metrics_dl = evaluate_all_metrics(y_val_real, val_preds, val_weights)

    print("\n==================================================")
    print("          РЕЗУЛЬТАТЫ ЭТАПА 5: DEEP LEARNING       ")
    print("==================================================")
    print(f"NWRMSLE: {metrics_dl['NWRMSLE']:.4f}")
    print(f"WAPE:    {metrics_dl['WAPE']:.4f}")
    print(f"MAE:     {metrics_dl['MAE']:.4f}")
    print("==================================================")

    # Сохраняем метрику для итогового отчета
    res_df = pd.DataFrame(
        [
            {
                "Model": "Deep Learning (Entity Embeddings + MLP)",
                **metrics_dl,
            }
        ]
    )
    res_df.to_csv(config.RESULTS_DIR / "dl_metrics.csv", index=False)


if __name__ == "__main__":
    run_dl()
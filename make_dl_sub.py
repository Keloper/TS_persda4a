# make_dl_submission.py
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
from src.data import load_exogenous_data, load_test_data, load_train_data
from src.features import prepare_dataset


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


# 2. Архитектура нейросети
class TimeSeriesEmbeddingNet(nn.Module):

    def __init__(
        self,
        cat_dims: list,
        emb_dims: list,
        num_conts: int,
        hidden_dims=[128, 64],
    ):
        super().__init__()
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
                    nn.Dropout(0.2),
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


# 3. Полный пайплайн генерации сабмита
def make_dl_submission():


    exo = load_exogenous_data()

    train_df = load_train_data(start_date="2017-06-01")
    test_df = load_test_data()
    test_df[config.TARGET_COL] = 0.0

    test_ids = test_df[config.ID_COL].values

    print("\nГенерация признаков для Train + Test...")
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    del train_df, test_df
    gc.collect()

    combined_df, feature_cols = prepare_dataset(combined_df, exo)

    train_mask = combined_df[config.DATE_COL] <= "2017-08-15"
    test_mask = combined_df[config.DATE_COL] >= "2017-08-16"

    train_part = combined_df.loc[train_mask].copy()
    test_part = combined_df.loc[test_mask].copy()
    del combined_df
    gc.collect()

    cat_cols = ["store_nbr", "family", "type", "cluster"]
    cont_cols = [c for c in feature_cols if c not in cat_cols and c != "item_nbr"]

    cat_maps = {
        col: {val: i for i, val in enumerate(train_part[col].unique())}
        for col in cat_cols
    }
    cats_train = np.column_stack(
        [train_part[c].map(cat_maps[c]).fillna(0).values for c in cat_cols]
    )
    cats_test = np.column_stack(
        [test_part[c].map(cat_maps[c]).fillna(0).values for c in cat_cols]
    )

    means = train_part[cont_cols].mean()
    stds = train_part[cont_cols].std().replace(0, 1)

    conts_train = ((train_part[cont_cols] - means) / stds).values.astype(
        np.float32
    )
    conts_test = ((test_part[cont_cols] - means) / stds).values.astype(
        np.float32
    )

    # Таргет и веса
    y_train_log = np.log1p(train_part[config.TARGET_COL].values)
    train_weights = train_part["item_nbr"].map(
        exo["items"].set_index("item_nbr")["weight"]
    ).fillna(1.0).values

    batch_size = 4096
    train_loader = DataLoader(
        TabularTSDataset(cats_train, conts_train, y_train_log, train_weights),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        TabularTSDataset(cats_test, conts_test),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    # 5. Инициализация и обучение модели
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Обучение нейросети")

    cat_dims = [len(cat_maps[c]) + 1 for c in cat_cols]
    emb_dims = [min(16, max(4, (c + 1) // 2)) for c in cat_dims]

    model = TimeSeriesEmbeddingNet(
        cat_dims, emb_dims, len(cont_cols), hidden_dims=[128, 64]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction="none")

    epochs = 5
    model.train()
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

        print(f"Эпоха {epoch+1}/{epochs} - Weighted MSE: {total_loss / len(train_loader):.4f}")

    print("\nИнференс нейросети на тесте...")
    model.eval()
    preds_list = []
    with torch.no_grad():
        for b_cats, b_conts in test_loader:
            b_cats, b_conts = b_cats.to(device), b_conts.to(device)
            preds = model(b_cats, b_conts)
            preds_list.append(preds.cpu().numpy())

    preds_log = np.concatenate(preds_list)
    preds_sales = np.clip(np.expm1(preds_log), 0, None)

    # Если скользящее среднее продаж за неделю было нулевым — ставим 0
    zero_mask = test_part["sales_roll_mean_7"].values == 0
    preds_sales[zero_mask] = 0.0
    preds_sales[preds_sales < 0.05] = 0.0

    # Сохранение сабмита
    sub = pd.DataFrame({config.ID_COL: test_ids, config.TARGET_COL: preds_sales})
    sub_path_root = config.BASE_DIR / "submission.csv"
    sub_path_dl = config.PREDICTIONS_DIR / "submission_dl.csv"

    sub.to_csv(sub_path_root, index=False)
    sub.to_csv(sub_path_dl, index=False)


    print(f"Всего строк: {len(sub):,}")
    print(sub.head(10))


if __name__ == "__main__":
    make_dl_submission()
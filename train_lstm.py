"""
train_lstm.py
---------------
Optional ADVANCED model referenced in the Project Proposal's Conceptual
Design ("sequence models (LSTM/GRU) that capture temporal deterioration
patterns"). This is a separate, optional stage — the core pipeline
(train.py) already works fully without it using Logistic Regression /
Random Forest / XGBoost.

Requires: pip install torch

Run: python train_lstm.py
"""

import json
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    torch = None

from src.preprocessing import VITAL_COLS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SEQ_LEN = 10  # days of raw vitals fed into the LSTM per sample
HORIZON = 6


def build_sequences(df: pd.DataFrame, seq_len=SEQ_LEN, horizon=HORIZON):
    """Builds (sequence, label) pairs directly from raw daily vitals,
    instead of the hand-engineered window features used by the baseline
    models — the LSTM is expected to learn temporal patterns itself."""
    X, y, meta = [], [], []
    for pid, g in df.groupby("patient_id"):
        g = g.sort_values("day").reset_index(drop=True)
        onset_day = g["onset_day"].iloc[0]
        has_event = not pd.isna(onset_day)
        vitals = g[VITAL_COLS].values

        for end_idx in range(seq_len, len(g) + 1):
            end_day = g["day"].iloc[end_idx - 1]
            seq = vitals[end_idx - seq_len:end_idx]
            if has_event and end_day < onset_day <= end_day + horizon:
                label = 1
                days_to_onset = int(onset_day - end_day)
            else:
                label = 0
                days_to_onset = np.nan
            X.append(seq)
            y.append(label)
            meta.append({"patient_id": pid, "window_end_day": end_day, "days_to_onset": days_to_onset})

    return np.array(X), np.array(y), pd.DataFrame(meta)


class VitalsLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits.squeeze(-1)


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():
    if torch is None:
        print("PyTorch is not installed. Run: pip install torch")
        return

    df = pd.read_csv("data/processed/vitals_clean.csv")
    X, y, meta = build_sequences(df)
    print(f"Built {len(X)} sequences, positive rate: {y.mean():.3f}")

    patient_ids = meta["patient_id"].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.25, random_state=42)
    train_mask = meta["patient_id"].isin(train_ids).values
    test_mask = ~train_mask

    n_features = X.shape[-1]
    scaler = StandardScaler().fit(X[train_mask].reshape(-1, n_features))
    X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape)

    train_ds = SequenceDataset(X_scaled[train_mask], y[train_mask])
    test_ds = SequenceDataset(X_scaled[test_mask], y[test_mask])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = VitalsLSTM(n_features=n_features)
    pos_weight = torch.tensor([(len(y[train_mask]) - y[train_mask].sum()) / max(1, y[train_mask].sum())])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 15
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch + 1}/{n_epochs} - loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_logits = model(test_ds.X)
        test_proba = torch.sigmoid(test_logits).numpy()
    auroc = roc_auc_score(y[test_mask], test_proba)
    print(f"\nLSTM Test AUROC: {auroc:.3f}")

    torch.save(model.state_dict(), "models/lstm_model.pt")
    with open("models/lstm_meta.json", "w") as f:
        json.dump({"seq_len": SEQ_LEN, "n_features": n_features, "test_auroc": auroc}, f)
    print("Saved LSTM model -> models/lstm_model.pt")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Hibridne arhitekture za predikciju koje kombinuju deep learning i klasične time-series modele.

6. TCN + LSTM Hybrid for Irregular Sampling (Irregular ICU Data)

Temporal Convolutional Networks (TCNs)
"""



import torch
import torch.nn as nn
import numpy as np

class IrregularTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, num_levels=4):
        super().__init__()
        # Dilated TCN: each layer doubles dilation (1, 2, 4, 8)
        self.tcn_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        dilation = 1
        for i in range(num_levels):
            # Causal convolution with dilation
            conv = nn.Conv1d(
                in_channels=input_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation  # Causal padding
            )
            self.tcn_layers.append(conv)
            self.residual_projections.append(
                nn.Conv1d(input_dim if i == 0 else hidden_dim, hidden_dim, 1)
            )
            dilation *= 2
            
        # Gated activation for adaptive feature selection
        self.gate = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, time_deltas):
        """
        Args:
            x: [batch, seq_len, features]
            time_deltas: [batch, seq_len] time gaps between observations
        """
        x = x.permute(0, 2, 1)  # [batch, features, seq_len]
        
        for conv, residual_projection in zip(self.tcn_layers, self.residual_projections):
            residual = x
            # Adjust convolution weights based on time deltas (simplified)
            conv_out = conv(x)
            conv_out = conv_out[:, :, :residual.size(2)]
            residual = residual_projection(residual)
            
            # Gate modulates output based on temporal sparsity
            # Larger gaps -> lower gate values
            gate_values = self.gate(conv_out)
            x = torch.tanh(conv_out) * gate_values + residual
            
        return x.permute(0, 2, 1)  # Back to [batch, seq_len, hidden]

class TCN_LSTM_Hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        # TCN for local irregular pattern extraction
        self.tcn = IrregularTCN(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            kernel_size=config['kernel_size']
        )
        
        # LSTM for long-term memory across gaps
        self.lstm = nn.LSTM(
            input_size=config['hidden_dim'],
            hidden_size=config['hidden_dim'],
            num_layers=1,
            batch_first=True
        )
        
        # Temporal masking: explicitly zero-out missing values
        self.mask_embedding = nn.Embedding(2, 1)  # 0=missing, 1=observed
        
        # Forecast head
        self.forecast_head = nn.Linear(config['hidden_dim'], config['forecast_len'])
        
    def forward(self, values, timestamps):
        """
        Args:
            values: [batch, seq_len, features] with NaNs for missing
            timestamps: Absolute timestamps [batch, seq_len]
        """
        # Create mask: 1 if observed, 0 if missing
        mask = (~torch.isnan(values)).long()
        
        # Replace NaNs with zeros and add mask as feature
        x = torch.nan_to_num(values, nan=0.0)
        mask_emb = self.mask_embedding(mask).squeeze(-1)
        x = x + mask_emb  # Add mask information
        
        # Compute time deltas between consecutive observations
        time_deltas = timestamps[:, 1:] - timestamps[:, :-1]
        time_deltas = torch.cat([torch.zeros_like(time_deltas[:, :1]), time_deltas], dim=1)
        time_deltas = time_deltas / 3600  # Convert to hours
        
        # TCN processes irregularly sampled data
        tcn_out = self.tcn(x, time_deltas)
        
        # LSTM captures long-term dependencies across gaps
        lstm_out, _ = self.lstm(tcn_out)
        
        # Predict future trajectory
        return self.forecast_head(lstm_out[:, -1, :])

# =========================
# Loto 7/39 adaptacija (loto7hh_4620_k41.csv) — demo izbačen
# =========================
import os

SEED = 39
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import copy
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


CSV_PATH = "/loto7hh_4620_k41.csv"
OUT_TXT = Path("/6_TCN_loto_v2_predikcija.txt")

N_MIN, N_MAX = 1, 39
K = 7
LOOK_BACK = 128
WINDOWS_RF = (20, 50, 100)
BACKTEST_N = 100
VAL_N = 200
EPOCHS = 30
BATCH = 64
LR = 1e-3
HIDDEN_DIM = 64
KERNEL_SIZE = 3

T0 = time.time()
print()
print("START 6_TCN_loto_v2", datetime.today())
print()

df = pd.read_csv(CSV_PATH).iloc[:, :K].astype(int)
draws = np.sort(df.values, axis=1)
N_total = draws.shape[0]
if not ((draws >= N_MIN) & (draws <= N_MAX)).all():
    raise ValueError("CSV ima brojeve van opsega 1..39.")
for idx, row in enumerate(draws):
    if len(set(row.tolist())) != K:
        raise ValueError(f"Red {idx} nema 7 jedinstvenih brojeva: {row.tolist()}")

print(f"CSV: {CSV_PATH}")
print(f"Broj izvlačenja: {N_total}, brojeva po kolu: {K}")
print()


def draws_to_multihot(rows):
    out = np.zeros((rows.shape[0], N_MAX), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, row - 1] = 1.0
    return out


def rolling_features(y_multi):
    cum = np.cumsum(y_multi, axis=0)
    blocks = []
    for w in WINDOWS_RF:
        rolled = np.zeros_like(cum, dtype=np.float32)
        rolled[1:w + 1] = cum[:w]
        rolled[w + 1:] = cum[w:-1] - cum[:-w - 1]
        blocks.append(rolled / float(w))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def gap_matrix(rows):
    n = rows.shape[0]
    gap = np.zeros((n, N_MAX), dtype=np.float32)
    last_seen = np.full(N_MAX, -1, dtype=int)
    for i, row in enumerate(rows):
        for k in range(N_MAX):
            gap[i, k] = (i - last_seen[k]) if last_seen[k] >= 0 else i + 1
        for v in row:
            last_seen[v - 1] = i
    return gap


def make_sequences(features, targets, look_back):
    X, Y, T = [], [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back:i])
        Y.append(targets[i])
        T.append(np.arange(i - look_back, i, dtype=np.float32) * 3600.0)
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(Y, dtype=np.float32),
        np.asarray(T, dtype=np.float32),
    )


def topk_from_scores(scores_1d, k=K):
    s = np.asarray(scores_1d, dtype=float)
    order = np.lexsort((np.arange(N_MAX), -s))
    return np.sort(order[:k] + 1)


def avg_hits(scores_2d, y_true):
    hits = 0
    for i in range(scores_2d.shape[0]):
        true_set = set(np.where(y_true[i] == 1)[0] + 1)
        pred_set = set(topk_from_scores(scores_2d[i]).tolist())
        hits += len(true_set & pred_set)
    return hits / scores_2d.shape[0]


def safe_auc(y_true, scores):
    try:
        return roc_auc_score(y_true, scores, average="macro")
    except Exception:
        return float("nan")


def safe_lrap(y_true, scores):
    try:
        return label_ranking_average_precision_score(y_true.astype(int), scores)
    except Exception:
        return float("nan")


def describe(pick):
    return (
        f"suma={int(pick.sum())}, "
        f"neparnih={int((pick % 2 == 1).sum())}/{K}, "
        f"niskih(<=19)={int((pick <= 19).sum())}/{K}, "
        f"raspon={int(pick.max() - pick.min())}"
    )


Y_full = draws_to_multihot(draws)
rolling_raw = rolling_features(Y_full)
gap_raw = gap_matrix(draws)

sum_col = draws.sum(axis=1, keepdims=True).astype(np.float32)
odd_col = (draws % 2 == 1).sum(axis=1, keepdims=True).astype(np.float32)
low_col = (draws <= 19).sum(axis=1, keepdims=True).astype(np.float32)
range_col = (draws.max(axis=1, keepdims=True) - draws.min(axis=1, keepdims=True)).astype(np.float32)
stats_raw = np.concatenate([sum_col, odd_col, low_col, range_col], axis=1)

step_features_raw = np.concatenate([Y_full, rolling_raw, gap_raw, stats_raw], axis=1).astype(np.float32)

START = max(LOOK_BACK, max(WINDOWS_RF))
feature_scaler = StandardScaler()
step_features = step_features_raw.copy()
step_features[START:] = feature_scaler.fit_transform(step_features_raw[START:]).astype(np.float32)
step_features[:START] = feature_scaler.transform(step_features_raw[:START]).astype(np.float32)

X_seq, Y_seq, T_seq = make_sequences(step_features, Y_full, LOOK_BACK)
X_seq = X_seq[START - LOOK_BACK:]
Y_seq = Y_seq[START - LOOK_BACK:]
T_seq = T_seq[START - LOOK_BACK:]

n_total = X_seq.shape[0]
n_train = n_total - BACKTEST_N
assert n_train > VAL_N + 200, "Premalo podataka za train/val/back-test."

X_tr, Y_tr, T_tr = X_seq[:n_train - VAL_N], Y_seq[:n_train - VAL_N], T_seq[:n_train - VAL_N]
X_val, Y_val, T_val = X_seq[n_train - VAL_N:n_train], Y_seq[n_train - VAL_N:n_train], T_seq[n_train - VAL_N:n_train]
X_back, Y_back, T_back = X_seq[n_train:], Y_seq[n_train:], T_seq[n_train:]
X_next = step_features[-LOOK_BACK:].reshape(1, LOOK_BACK, step_features.shape[1]).astype(np.float32)
T_next = (np.arange(N_total - LOOK_BACK, N_total, dtype=np.float32) * 3600.0).reshape(1, LOOK_BACK)

INPUT_DIM = X_seq.shape[-1]
print(f"Feature dim: {INPUT_DIM}, LOOK_BACK: {LOOK_BACK}")
print(f"Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Back-test: {X_back.shape[0]}")
print()


config = {
    'input_dim': INPUT_DIM,
    'hidden_dim': HIDDEN_DIM,
    'kernel_size': KERNEL_SIZE,
    'forecast_len': N_MAX  # 39 sigmoid logita po broju 1..39
}

model = TCN_LSTM_Hybrid(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

pos_weight_value = (N_MAX - K) / K
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((N_MAX,), pos_weight_value, dtype=torch.float32))


def make_loader(X, T, Y, shuffle):
    generator = torch.Generator()
    generator.manual_seed(SEED)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(T), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=generator)


train_loader = make_loader(X_tr, T_tr, Y_tr, shuffle=False)
val_X_t = torch.from_numpy(X_val)
val_T_t = torch.from_numpy(T_val)
val_Y_t = torch.from_numpy(Y_val)

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
best_epoch = 0

print("Treniranje TCN+LSTM na loto podacima ...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    seen = 0
    for xb, tb, yb in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb, tb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += float(loss.detach().cpu()) * xb.size(0)
        seen += xb.size(0)
    train_loss /= max(seen, 1)

    model.eval()
    with torch.no_grad():
        val_logits = model(val_X_t, val_T_t)
        val_loss = float(criterion(val_logits, val_Y_t).detach().cpu())
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

    if epoch == 1 or epoch % 50 == 0 or epoch == EPOCHS:
        print(f"epoch {epoch:4d}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  best_epoch={best_epoch}")

final_state = copy.deepcopy(model.state_dict())
print()
print(f"✅ Trening završen. best_epoch={best_epoch}, best_val_loss={best_val_loss:.5f}")
print()


def predict_scores(model, X, T):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, X.shape[0], BATCH):
            xb = torch.from_numpy(X[s:s + BATCH])
            tb = torch.from_numpy(T[s:s + BATCH])
            logits = model(xb, tb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(out)


def evaluate(model, X, T, Y):
    scores = predict_scores(model, X, T)
    return scores, avg_hits(scores, Y), safe_auc(Y, scores), safe_lrap(Y, scores)


model.load_state_dict(best_state)
scores_best, h_best, auc_best, lrap_best = evaluate(model, X_back, T_back, Y_back)
next_best = predict_scores(model, X_next, T_next)[0]
pick_best = topk_from_scores(next_best)

model.load_state_dict(final_state)
scores_final, h_final, auc_final, lrap_final = evaluate(model, X_back, T_back, Y_back)
next_final = predict_scores(model, X_next, T_next)[0]
pick_final = topk_from_scores(next_final)

ensemble_scores = (scores_best + scores_final) / 2.0
h_ens = avg_hits(ensemble_scores, Y_back)
auc_ens = safe_auc(Y_back, ensemble_scores)
lrap_ens = safe_lrap(Y_back, ensemble_scores)
pick_ens = topk_from_scores((next_best + next_final) / 2.0)

for name, pick in [("TCN_LSTM_best", pick_best), ("TCN_LSTM_final", pick_final), ("TCN_LSTM_ensemble", pick_ens)]:
    assert len(set(pick.tolist())) == K, f"{name} nema 7 jedinstvenih brojeva"
    assert pick.min() >= N_MIN and pick.max() <= N_MAX, f"{name} van opsega"
    assert list(pick) == sorted(pick.tolist()), f"{name} nije sortiran"

print("Predikcija sledeće Loto 7/39 kombinacije:")
print(f"TCN_LSTM_best     -> {pick_best.tolist()}  ({describe(pick_best)})")
print(f"TCN_LSTM_final    -> {pick_final.tolist()}  ({describe(pick_final)})")
print(f"TCN_LSTM_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})")
print()

print("Back-test (poslednjih 100 izvlačenja):")
print(f"{'model':<20} {'hits/7':>8} {'hit%':>7} {'AUC':>7} {'LRAP':>7}")
print(f"{'TCN_LSTM_best':<20} {h_best:>8.3f} {100*h_best/K:>6.1f}% {auc_best:>7.3f} {lrap_best:>7.3f}")
print(f"{'TCN_LSTM_final':<20} {h_final:>8.3f} {100*h_final/K:>6.1f}% {auc_final:>7.3f} {lrap_final:>7.3f}")
print(f"{'TCN_LSTM_ensemble':<20} {h_ens:>8.3f} {100*h_ens/K:>6.1f}% {auc_ens:>7.3f} {lrap_ens:>7.3f}")
print(f"(slučajan baseline ≈ {7*7/39:.3f} hits/7)")
print()


elapsed = time.time() - T0
with OUT_TXT.open("a", encoding="utf-8") as f:
    f.write(f"\n--- {datetime.today()} (seed={SEED}, N={N_total}, epochs={EPOCHS}) ---\n")
    f.write(f"TCN_LSTM_best     -> {pick_best.tolist()}  ({describe(pick_best)})\n")
    f.write(f"TCN_LSTM_final    -> {pick_final.tolist()}  ({describe(pick_final)})\n")
    f.write(f"TCN_LSTM_ensemble -> {pick_ens.tolist()}  ({describe(pick_ens)})\n")
    f.write(
        f"back-test: BEST hits/7={h_best:.3f}, AUC={auc_best:.3f}, LRAP={lrap_best:.3f}; "
        f"FINAL hits/7={h_final:.3f}, AUC={auc_final:.3f}, LRAP={lrap_final:.3f}; "
        f"ENSEMBLE hits/7={h_ens:.3f}, AUC={auc_ens:.3f}, LRAP={lrap_ens:.3f}; "
        f"baseline={7*7/39:.3f}\n"
    )
    f.write(f"elapsed={elapsed:.1f}s\n")

print(f"Snimljeno u: {OUT_TXT}")
print()
print("STOP", datetime.today())
print(f"Ukupno vreme: {str(timedelta(seconds=int(elapsed)))}  ({elapsed:.1f} s)")
print()



"""
START 6_TCN_loto_v2 2026-05-25 15:52:55.713818

CSV: /loto7hh_4620_k41.csv
Broj izvlačenja: 4620, brojeva po kolu: 7

Feature dim: 199, LOOK_BACK: 128
Train: 4192, Val: 200, Back-test: 100

Treniranje TCN+LSTM na loto podacima ...
epoch    1/30  train_loss=1.13850  val_loss=1.13729  best_epoch=1
epoch   30/30  train_loss=0.93776  val_loss=1.29130  best_epoch=1

✅ Trening završen. best_epoch=1, best_val_loss=1.13729

Predikcija sledeće Loto 7/39 kombinacije:
TCN_LSTM_best     -> [8, x, 11, y, 23, z, 34]  (suma=135, neparnih=3/7, niskih(<=19)=4/7, raspon=26)
TCN_LSTM_final    -> [10, x, 14, y, 29, z, 38]  (suma=158, neparnih=2/7, niskih(<=19)=4/7, raspon=28)
TCN_LSTM_ensemble -> [10, x, 18, y, 29, z, 38]  (suma=167, neparnih=3/7, niskih(<=19)=3/7, raspon=28)

Back-test (poslednjih 100 izvlačenja):
model                  hits/7    hit%     AUC    LRAP
TCN_LSTM_best           1.340   19.1%   0.500   0.257
TCN_LSTM_final          1.170   16.7%   0.494   0.249
TCN_LSTM_ensemble       1.220   17.4%   0.494   0.250
(slučajan baseline ≈ 1.256 hits/7)

Snimljeno u: /6_TCN_loto_v2_predikcija.txt

STOP 2026-05-25 16:12:15.273128
Ukupno vreme: 0:19:19  (1159.6 s)
"""

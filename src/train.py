"""Train and compare: rule baseline, Logistic Regression, Random Forest, LSTM.

Run:  python -m src.train
Writes: models/lstm_weights.npz, models/metrics.json
(PyTorch is only needed here, on the laptop - NOT in the deployed app.)
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from torch import nn

from .config import DATA_PROCESSED, MODELS_DIR
from .features import SEQ_FEATURES, window_to_tabular
from .models import (NumpyLSTM, SEED, build_all_windows, fit_sklearn_models, split_masks)

torch.manual_seed(SEED)
np.random.seed(SEED)


class LSTMNet(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1])).squeeze(-1)


def train_lstm(Xtr, ytr, Xva, yva, epochs=40):
    mean = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    std = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    f = lambda a: torch.tensor((a - mean) / std, dtype=torch.float32)
    Xtr_t, Xva_t = f(Xtr), f(Xva)
    ytr_t = torch.tensor(ytr)
    net = LSTMNet(Xtr.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, patience = -1, None, 0
    history = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        losses = []
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva_t)).numpy()
        auc = roc_auc_score(yva, pv)
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_auc": float(auc)})
        print(f"  epoch {ep + 1:2d}  val AUC {auc:.4f}")
        if auc > best:
            best, patience = auc, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    net.load_state_dict(best_state)
    return net, mean, std, history


def export_lstm(net, mean, std):
    sd = {k: v.numpy() for k, v in net.state_dict().items()}
    np.savez(MODELS_DIR / "lstm_weights.npz",
             w_ih=sd["lstm.weight_ih_l0"], w_hh=sd["lstm.weight_hh_l0"],
             b_ih=sd["lstm.bias_ih_l0"], b_hh=sd["lstm.bias_hh_l0"],
             fc_w=sd["fc.weight"], fc_b=sd["fc.bias"], mean=mean, std=std)


def evaluate(name, y_val, p_val, y_test, p_test, loc_test):
    # choose the alert threshold on VALIDATION data (never on test)
    grid = np.linspace(0.05, 0.95, 91)
    thr = grid[int(np.argmax([f1_score(y_val, p_val >= t, zero_division=0) for t in grid]))]
    pred = p_test >= thr
    fpr, tpr, _ = roc_curve(y_test, p_test)
    keep = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
    per_loc = {l: float(roc_auc_score(y_test[loc_test == l], p_test[loc_test == l]))
               for l in np.unique(loc_test) if 0 < y_test[loc_test == l].mean() < 1}
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "pr_auc": float(average_precision_score(y_test, p_test)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred)),
        "brier": float(brier_score_loss(y_test, np.clip(p_test, 0, 1))),
        "threshold": float(thr),
        "roc_curve": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "roc_auc_by_location": per_loc,
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    daily = pd.read_csv(DATA_PROCESSED / "daily.csv")
    X, y, dates, locs = build_all_windows(daily)
    tr, va, te = split_masks(dates)
    print(f"windows: train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}   "
          f"positive rate train {y[tr].mean():.1%} / test {y[te].mean():.1%}")

    results = []

    # 0) naive baseline: "an infection period just happened -> risk stays high"
    Xt = window_to_tabular(X)
    names_idx = {n: i for i, n in enumerate(__import__("src.features", fromlist=["x"]).tabular_feature_names())}
    base = Xt[:, names_idx["hutton_days_last14"]] / 14.0
    results.append(evaluate("Baseline: recent-days rule", y[va], base[va], y[te], base[te], locs[te]))

    # 1+2) sklearn models
    lr, rf = fit_sklearn_models(X[tr], y[tr])
    for name, m in (("Logistic Regression", lr), ("Random Forest", rf)):
        p = m.predict_proba(Xt)[:, 1]
        results.append(evaluate(name, y[va], p[va], y[te], p[te], locs[te]))

    # 3) LSTM
    print("Training LSTM ...")
    net, mean, std, history = train_lstm(X[tr], y[tr], X[va], y[va])
    export_lstm(net, mean, std)
    p_lstm = NumpyLSTM().predict_proba(X)          # NumPy version = what the app will use
    with torch.no_grad():
        p_torch = torch.sigmoid(net(torch.tensor((X[:512] - mean) / std, dtype=torch.float32))).numpy()
    assert np.allclose(p_torch, p_lstm[:512], atol=1e-4), "NumPy LSTM != PyTorch LSTM"
    results.append(evaluate("LSTM (deep learning)", y[va], p_lstm[va], y[te], p_lstm[te], locs[te]))

    out = {
        "split": {"train": "2015-2021", "validation": "2022", "test": "2023-2024"},
        "n_windows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "positive_rate_test": float(y[te].mean()),
        "results": results,
        "lstm_history": history,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(out, f, indent=1)

    print("\nTEST SET (2023-2024, never seen during training)")
    print(f"{'model':32s} ROC-AUC  PR-AUC     F1  precision  recall  brier")
    for r in results:
        print(f"{r['model']:32s} {r['roc_auc']:.3f}   {r['pr_auc']:.3f}  {r['f1']:.3f}   {r['precision']:.3f}     {r['recall']:.3f}   {r['brier']:.3f}")


if __name__ == "__main__":
    main()

"""Model helpers shared by training and the web app.

- Logistic Regression + Random Forest are cheap, so the app re-fits them at start-up
  from the bundled data (no pickle/version problems on the host).
- The LSTM is trained with PyTorch on a laptop, then exported to a .npz file and run
  with plain NumPy here, so the deployed app does not need PyTorch at all.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LOCATIONS, MODELS_DIR, TRAIN_END, VAL_END
from .features import make_windows, window_to_tabular

SEED = 42


# ---------------------------------------------------------------- dataset
def build_all_windows(daily: pd.DataFrame):
    """All locations -> (X_seq, y, dates, location_names)."""
    Xs, ys, ds, ls = [], [], [], []
    for name in LOCATIONS:
        d = daily[daily.location == name].drop(columns="location").copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date").sort_index()
        X, y, dates = make_windows(d)
        Xs.append(X); ys.append(y); ds.append(dates); ls += [name] * len(y)
    return (np.concatenate(Xs), np.concatenate(ys),
            pd.DatetimeIndex(np.concatenate([d.values for d in ds])), np.array(ls))


def split_masks(dates: pd.DatetimeIndex):
    train = dates <= pd.Timestamp(TRAIN_END)
    val = (dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VAL_END))
    test = dates > pd.Timestamp(VAL_END)
    return train, val, test


# ---------------------------------------------------------------- sklearn models
def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, random_state=SEED))


def make_forest():
    return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                  n_jobs=-1, random_state=SEED)


def fit_sklearn_models(X_seq_train: np.ndarray, y_train: np.ndarray):
    Xt = window_to_tabular(X_seq_train)
    lr = make_logreg().fit(Xt, y_train)
    rf = make_forest().fit(Xt, y_train)
    return lr, rf


# ---------------------------------------------------------------- NumPy LSTM
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyLSTM:
    """Inference-only LSTM (1 layer) + linear head. Weights exported from PyTorch."""

    def __init__(self, path=MODELS_DIR / "lstm_weights.npz"):
        w = np.load(path)
        self.w_ih, self.w_hh = w["w_ih"], w["w_hh"]
        self.b = w["b_ih"] + w["b_hh"]
        self.fc_w, self.fc_b = w["fc_w"], w["fc_b"]
        self.mean, self.std = w["mean"], w["std"]
        self.hidden = self.w_hh.shape[1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = (X - self.mean) / self.std
        n, steps, _ = X.shape
        h = np.zeros((n, self.hidden)); c = np.zeros((n, self.hidden))
        H = self.hidden
        for t in range(steps):
            z = X[:, t] @ self.w_ih.T + h @ self.w_hh.T + self.b
            i, f, g, o = (_sigmoid(z[:, :H]), _sigmoid(z[:, H:2 * H]),
                          np.tanh(z[:, 2 * H:3 * H]), _sigmoid(z[:, 3 * H:]))
            c = f * c + i * g
            h = o * np.tanh(c)
        return _sigmoid(h @ self.fc_w.T + self.fc_b).ravel()


def load_metrics() -> dict:
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)

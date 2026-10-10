"""Turn hourly weather into daily features, 14-day windows and a 5-day-ahead risk label.

The SAME functions are used for training and for the live app, so there is no mismatch.
"""
import numpy as np
import pandas as pd

from .config import (HORIZON_DAYS, HUTTON_MAX_TEMP, HUTTON_MIN_RH_HOURS, HUTTON_MIN_TEMP,
                     RH_THRESHOLD, WINDOW_DAYS)

# Columns produced per day (order matters: it is the LSTM input order).
DAILY_COLS = [
    "t_mean", "t_min", "t_max",
    "rh_mean", "rh_min", "hrs_rh90", "hrs_favourable",
    "rain_sum", "hrs_rain", "dew_depression", "wind_mean",
    "cloud_mean", "radiation_sum", "hutton_day",
]
SEQ_FEATURES = DAILY_COLS + ["doy_sin", "doy_cos"]

# Human-friendly names (used in SHAP charts for the farmer-facing app)
PRETTY = {
    "t_mean": "Avg temperature", "t_min": "Night (min) temperature", "t_max": "Day (max) temperature",
    "rh_mean": "Avg humidity", "rh_min": "Lowest humidity", "hrs_rh90": "Hours of very high humidity (>=90%)",
    "hrs_favourable": "Hours humid AND cool (blight-friendly)", "rain_sum": "Rainfall",
    "hrs_rain": "Hours of rain", "dew_depression": "Air dryness (temp - dew point)",
    "wind_mean": "Wind speed", "cloud_mean": "Cloud cover", "radiation_sum": "Sunshine energy",
    "hutton_day": "Blight-favourable day", "doy_sin": "Season (sin)", "doy_cos": "Season (cos)",
}


def hourly_to_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather (columns: time + HOURLY_VARS) -> one row per day."""
    h = h.copy()
    h["time"] = pd.to_datetime(h["time"])
    h["date"] = h["time"].dt.normalize()
    t, rh = h["temperature_2m"], h["relative_humidity_2m"]
    h["_rh90"] = (rh >= RH_THRESHOLD).astype(float)
    h["_fav"] = ((rh >= RH_THRESHOLD) & (t >= 10) & (t <= 25)).astype(float)
    h["_rain_h"] = (h["precipitation"] > 0.1).astype(float)
    h["_dep"] = t - h["dew_point_2m"]

    g = h.groupby("date")
    d = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(),
        "t_min": g["temperature_2m"].min(),
        "t_max": g["temperature_2m"].max(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "rh_min": g["relative_humidity_2m"].min(),
        "hrs_rh90": g["_rh90"].sum(),
        "hrs_favourable": g["_fav"].sum(),
        "rain_sum": g["precipitation"].sum(),
        "hrs_rain": g["_rain_h"].sum(),
        "dew_depression": g["_dep"].mean(),
        "wind_mean": g["wind_speed_10m"].mean(),
        "cloud_mean": g["cloud_cover"].mean(),
        "radiation_sum": g["shortwave_radiation"].sum() / 1000.0,
    })
    # keep only complete days (24 hourly rows)
    d = d[g.size() >= 24].copy()
    d["hutton_day"] = ((d["t_min"] >= HUTTON_MIN_TEMP) & (d["t_max"] <= HUTTON_MAX_TEMP)
                       & (d["hrs_rh90"] >= HUTTON_MIN_RH_HOURS)).astype(float)
    d.index.name = "date"
    return d.interpolate(limit=2).ffill().bfill()


def add_season(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    doy = d.index.dayofyear.values
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return d


def hutton_period_end(d: pd.DataFrame) -> pd.Series:
    """True on a day when it AND the previous day were both blight-favourable (Smith-style infection period)."""
    hd = d["hutton_day"] > 0
    return hd & hd.shift(1, fill_value=False)


def make_windows(d: pd.DataFrame, with_labels: bool = True):
    """Sliding windows over one location's daily table.

    X[i]  = the WINDOW_DAYS days ending on day t        (what we know on day t)
    y[i]  = 1 if an infection period happens fully inside days t+1 .. t+HORIZON  (future only)
    Returns X (N, WINDOW, F), y (N,), dates (N,) - dates are the 'forecast made on' day t.
    """
    d = add_season(d)
    arr = d[SEQ_FEATURES].to_numpy(dtype=np.float32)
    n = len(d)
    ends = hutton_period_end(d).to_numpy()
    X, y, dates = [], [], []
    last = n - HORIZON_DAYS if with_labels else n
    for t in range(WINDOW_DAYS - 1, last):
        X.append(arr[t - WINDOW_DAYS + 1: t + 1])
        dates.append(d.index[t])
        if with_labels:
            # a period "ending" on day k uses days k-1 and k; require k-1 >= t+1 -> k in t+2..t+H
            y.append(float(ends[t + 2: t + HORIZON_DAYS + 1].any()))
    X = np.stack(X)
    return (X, np.array(y, dtype=np.float32) if with_labels else None, pd.DatetimeIndex(dates))


# ---------- tabular features for Logistic Regression / Random Forest -----------
_KEY = ["t_mean", "t_min", "t_max", "rh_mean", "hrs_rh90", "hrs_favourable",
        "rain_sum", "hrs_rain", "dew_depression", "wind_mean", "cloud_mean", "radiation_sum", "hutton_day"]


def tabular_feature_names() -> list[str]:
    names = []
    for k in _KEY:
        names += [f"{k}__today", f"{k}__avg3d", f"{k}__avg7d", f"{k}__avg14d"]
    names += ["rh_min__avg3d", "hutton_days_last14", "hutton_streak_now", "doy_sin", "doy_cos"]
    return names


def window_to_tabular(X: np.ndarray) -> np.ndarray:
    """(N, WINDOW, F) -> (N, n_features). Summaries of the window (today, 3d, 7d, 14d means)."""
    idx = {c: i for i, c in enumerate(SEQ_FEATURES)}
    cols = []
    for k in _KEY:
        s = X[:, :, idx[k]]
        cols += [s[:, -1], s[:, -3:].mean(1), s[:, -7:].mean(1), s[:, -14:].mean(1)]
    cols.append(X[:, -3:, idx["rh_min"]].mean(1))
    hd = X[:, :, idx["hutton_day"]]
    cols.append(hd.sum(1))
    streak = np.zeros(len(X))
    alive = np.ones(len(X), dtype=bool)
    for j in range(X.shape[1] - 1, -1, -1):
        alive &= hd[:, j] > 0
        streak += alive
    cols.append(streak)
    cols.append(X[:, -1, idx["doy_sin"]])
    cols.append(X[:, -1, idx["doy_cos"]])
    return np.stack(cols, axis=1).astype(np.float32)

"""Upload handling + the 'lite' model that works from plain DAILY weather.

Two kinds of uploads are supported:
  * HOURLY weather (needs time, temperature, humidity)  -> full model (uses hours of high humidity)
  * DAILY weather  (needs date, min/max temp, humidity)  -> lite model (daily numbers only)
Optionally a daily file may contain a `blight_observed` (0/1) column: used to check / train on real records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import HORIZON_DAYS, WINDOW_DAYS
from .features import SEQ_FEATURES, add_season, hourly_to_daily

LITE_COLS = ["t_min", "t_max", "t_mean", "rh_mean", "rain_sum"]
LITE_SEQ = LITE_COLS + ["doy_sin", "doy_cos"]
LITE_IDX = [SEQ_FEATURES.index(c) for c in LITE_SEQ]

LITE_PRETTY = {"t_min": "Night (min) temperature", "t_max": "Day (max) temperature", "t_mean": "Avg temperature",
               "rh_mean": "Avg humidity", "rain_sum": "Rainfall", "doy_sin": "Time of year", "doy_cos": "Time of year"}

_ALIASES = {
    "time": ["date", "datetime", "time", "timestamp", "day", "date_time"],
    "temp": ["temperature_2m", "temperature", "temp", "t", "air_temperature", "temp_c", "temperature_c"],
    "tmin": ["t_min", "tmin", "temp_min", "min_temp", "temperature_min", "min_temperature", "minimum_temperature",
             "temperature_2m_min"],
    "tmax": ["t_max", "tmax", "temp_max", "max_temp", "temperature_max", "max_temperature", "maximum_temperature",
             "temperature_2m_max"],
    "tmean": ["t_mean", "tmean", "temp_mean", "avg_temp", "mean_temp", "temperature_mean", "temperature_2m_mean"],
    "rh": ["relative_humidity_2m", "relative_humidity", "humidity", "rh", "rh_mean", "humidity_mean", "avg_humidity",
           "mean_humidity", "relative_humidity_2m_mean", "humidity_percent"],
    "rhmin": ["rh_min", "humidity_min", "min_humidity", "relative_humidity_2m_min"],
    "rhmax": ["rh_max", "humidity_max", "max_humidity", "relative_humidity_2m_max"],
    "rain": ["precipitation", "rain", "rainfall", "rain_mm", "precip", "precipitation_sum", "rain_sum", "rainfall_mm"],
    "dew": ["dew_point_2m", "dew_point", "dewpoint", "dew_point_c"],
    "wind": ["wind_speed_10m", "wind_speed", "wind", "windspeed"],
    "cloud": ["cloud_cover", "cloud", "clouds"],
    "rad": ["shortwave_radiation", "radiation", "solar_radiation"],
    "blight": ["blight_observed", "blight", "disease", "outbreak", "disease_observed", "blight_present"],
}


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def _find(df: pd.DataFrame, key: str):
    cols = {_norm(c): c for c in df.columns}
    for a in _ALIASES[key]:
        if a in cols:
            return cols[a]
    return None


@dataclass
class Upload:
    kind: str = ""                      # "hourly" | "daily"
    daily: pd.DataFrame | None = None   # indexed by date. hourly -> all DAILY_COLS, daily -> LITE_COLS
    blight: pd.Series | None = None     # daily 0/1 observed blight (if provided)
    notes: list = field(default_factory=list)
    error: str = ""


def _magnus_dew(t: pd.Series, rh: pd.Series) -> pd.Series:
    a, b = 17.62, 243.12
    g = np.log(np.clip(rh, 1, 100) / 100.0) + a * t / (b + t)
    return b * g / (a - g)


def parse_upload(df: pd.DataFrame) -> Upload:
    up = Upload()
    if df is None or len(df) < 20:
        up.error = "The file has too few rows. Please upload at least ~3 weeks of weather."
        return up
    c_time = _find(df, "time")
    if c_time is None:
        up.error = ("I couldn't find a date column. Name it `date` (daily file) or `time` (hourly file), "
                    "e.g. 2024-08-15 or 2024-08-15 14:00.")
        return up
    t = pd.to_datetime(df[c_time], errors="coerce", dayfirst=False)
    if t.isna().mean() > 0.2:
        t2 = pd.to_datetime(df[c_time], errors="coerce", dayfirst=True)
        t = t2 if t2.isna().mean() < t.isna().mean() else t
    if t.isna().mean() > 0.2:
        up.error = f"I couldn't read the dates in column `{c_time}`. Use a format like 2024-08-15."
        return up
    df = df.assign(_t=t).dropna(subset=["_t"]).sort_values("_t").drop_duplicates("_t")
    spacing = df["_t"].diff().dropna().median()
    hourly = spacing <= pd.Timedelta(hours=1.5)
    num = lambda col: pd.to_numeric(df[col], errors="coerce") if col else None

    if hourly:
        c_t, c_rh = _find(df, "temp"), _find(df, "rh")
        if not c_t or not c_rh:
            up.error = "Hourly file needs a temperature column (`temperature_2m`) and a humidity column (`relative_humidity_2m`)."
            return up
        h = pd.DataFrame({"time": df["_t"].values, "temperature_2m": num(c_t).values,
                          "relative_humidity_2m": num(c_rh).values})
        miss = []
        for key, name, default in (("dew", "dew_point_2m", None), ("rain", "precipitation", 0.0),
                                   ("wind", "wind_speed_10m", 8.0), ("cloud", "cloud_cover", 60.0),
                                   ("rad", "shortwave_radiation", 200.0)):
            col = _find(df, key)
            if col:
                h[name] = num(col).values
            elif key == "dew":
                h[name] = _magnus_dew(h["temperature_2m"], h["relative_humidity_2m"]).values
                miss.append("dew point (calculated)")
            else:
                h[name] = default
                miss.append(f"{name} (assumed typical value)")
        h = h.dropna(subset=["temperature_2m", "relative_humidity_2m"])
        daily = hourly_to_daily(h)
        if len(daily) < 15:
            up.error = f"Only {len(daily)} complete days found (each day needs 24 hourly rows). Need at least 15."
            return up
        up.kind, up.daily = "hourly", daily
        up.notes.append(f"Hourly file detected → {len(daily)} complete days. Using the **full model**.")
        if miss:
            up.notes.append("Columns not provided: " + ", ".join(miss) + ". These matter little to the prediction.")
        return up

    # ---- daily file
    c_tmin, c_tmax, c_tmean, c_t = _find(df, "tmin"), _find(df, "tmax"), _find(df, "tmean"), _find(df, "temp")
    c_rh, c_rhmin, c_rhmax, c_rain = _find(df, "rh"), _find(df, "rhmin"), _find(df, "rhmax"), _find(df, "rain")
    d = pd.DataFrame(index=pd.DatetimeIndex(df["_t"].dt.normalize()))
    d.index.name = "date"
    if c_tmin and c_tmax:
        d["t_min"], d["t_max"] = num(c_tmin).values, num(c_tmax).values
    elif c_tmean or c_t:
        up.error = ("Daily file needs both minimum and maximum temperature columns (`temp_min`, `temp_max`). "
                    "Night-time temperature matters for blight.")
        return up
    else:
        up.error = "Daily file needs `temp_min` and `temp_max` columns (°C)."
        return up
    d["t_mean"] = num(c_tmean).values if c_tmean else (d["t_min"] + d["t_max"]) / 2
    if c_rh:
        d["rh_mean"] = num(c_rh).values
    elif c_rhmin and c_rhmax:
        d["rh_mean"] = ((num(c_rhmin) + num(c_rhmax)) / 2).values
        up.notes.append("Humidity = average of min and max humidity.")
    else:
        up.error = "Daily file needs a humidity column (`humidity` or `rh_mean`, in %)."
        return up
    if c_rain:
        d["rain_sum"] = num(c_rain).values
    else:
        d["rain_sum"] = 0.0
        up.notes.append("No rainfall column - assumed 0 mm.")
    c_bl = _find(df, "blight")
    bl = pd.Series(num(c_bl).values, index=d.index) if c_bl else None
    d = d[~d.index.duplicated()]
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    gaps = len(full) - len(d)
    d = d.reindex(full).interpolate(limit=3).ffill(limit=3).dropna()
    d.index.name = "date"
    if gaps:
        up.notes.append(f"{gaps} missing days were filled by interpolation where short.")
    if len(d) < 15:
        up.error = f"Only {len(d)} usable days. Need at least 15 consecutive days."
        return up
    if d["rh_mean"].max() <= 1.5:
        d["rh_mean"] *= 100
        up.notes.append("Humidity looked like 0-1 fractions → converted to %.")
    if bl is not None:
        bl = bl[~bl.index.duplicated()].reindex(d.index).fillna(0).clip(0, 1)
        up.blight = bl
    up.kind, up.daily = "daily", d
    up.notes.append(f"Daily file detected → {len(d)} days. Using the **lite model** (daily numbers only).")
    return up


# ---------------------------------------------------------------- windows / features for lite model
def lite_windows(d: pd.DataFrame, labels: pd.Series | None = None):
    """Windows over a daily table with LITE_COLS. labels: daily 0/1 series -> y = any in next HORIZON days."""
    d = add_season(d)
    arr = d[LITE_SEQ].to_numpy(dtype=np.float32)
    lab = labels.reindex(d.index).fillna(0).to_numpy() if labels is not None else None
    X, y, dates = [], [], []
    last = len(d) - HORIZON_DAYS if labels is not None else len(d)
    for t in range(WINDOW_DAYS - 1, last):
        X.append(arr[t - WINDOW_DAYS + 1: t + 1])
        dates.append(d.index[t])
        if lab is not None:
            y.append(float(lab[t + 1: t + HORIZON_DAYS + 1].any()))
    if not X:
        return np.zeros((0, WINDOW_DAYS, len(LITE_SEQ)), np.float32), np.array([]), pd.DatetimeIndex([])
    return np.stack(X), (np.array(y, dtype=np.float32) if labels is not None else None), pd.DatetimeIndex(dates)


def lite_tabular_names() -> list[str]:
    names = []
    for k in LITE_COLS:
        names += [f"{k}__today", f"{k}__avg3d", f"{k}__avg7d", f"{k}__avg14d"]
    names += ["rh_mean__max3d", "rain_sum__sum7d", "doy_sin", "doy_cos"]
    return names


def lite_tabular(X: np.ndarray) -> np.ndarray:
    cols = []
    for i in range(len(LITE_COLS)):
        s = X[:, :, i]
        cols += [s[:, -1], s[:, -3:].mean(1), s[:, -7:].mean(1), s[:, -14:].mean(1)]
    rh = LITE_COLS.index("rh_mean"); rn = LITE_COLS.index("rain_sum")
    cols += [X[:, -3:, rh].max(1), X[:, -7:, rn].sum(1), X[:, -1, -2], X[:, -1, -1]]
    return np.stack(cols, axis=1).astype(np.float32)


def group_name(name: str, pretty: dict) -> str:
    base = name.split("__")[0]
    if base.startswith("hutton"):
        return "Recent blight-friendly days"
    if base.startswith("doy"):
        return "Time of year"
    return pretty.get(base, base)

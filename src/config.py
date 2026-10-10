"""Shared settings for the whole project (data, features, training and the app)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# Tomato-growing regions (cool/humid hills where late blight is common + warmer plains).
LOCATIONS = {
    "Ooty (Nilgiris, TN)":      {"lat": 11.41, "lon": 76.70, "note": "Cool, misty hills - classic blight region"},
    "Shimla (Himachal)":        {"lat": 31.10, "lon": 77.17, "note": "Cool hills, monsoon humidity"},
    "Darjeeling (W. Bengal)":   {"lat": 27.04, "lon": 88.26, "note": "Very humid hills"},
    "Kolar (Karnataka)":        {"lat": 13.14, "lon": 78.13, "note": "Major tomato belt"},
    "Nashik (Maharashtra)":     {"lat": 20.00, "lon": 73.79, "note": "Major tomato belt"},
    "Udaipur (Rajasthan)":      {"lat": 24.58, "lon": 73.68, "note": "Semi-arid, monsoon-driven risk"},
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Forecasting setup
WINDOW_DAYS = 14      # how many past days the model sees
HORIZON_DAYS = 5      # how many days ahead we forecast

# Time-based split (never train on the future)
TRAIN_END = "2021-12-31"
VAL_END = "2022-12-31"   # validation = 2022, test = 2023-2024

# Weak-label rule: Smith-period-style late blight infection rule (two consecutive favourable days).
# A "favourable day" = night temp >= 10C, day temp <= 27C, and >= 11 hours of RH >= 90%.
# (Column is still called `hutton_day` in the code for historical reasons.)
HUTTON_MIN_TEMP = 10.0     # deg C, daily minimum must be >= this
HUTTON_MAX_TEMP = 27.0     # deg C, daily maximum must be <= this
HUTTON_MIN_RH_HOURS = 11   # hours with RH >= 90% per day
RH_THRESHOLD = 90.0

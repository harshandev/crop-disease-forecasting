"""Live weather (free Open-Meteo forecast API, no key): last 14 days + next 5 days."""
from __future__ import annotations

import pandas as pd
import requests

from .config import HOURLY_VARS
from .features import hourly_to_daily

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_recent_and_forecast(lat: float, lon: float, timeout: int = 15):
    """Returns (daily_table, today) where daily_table covers ~15 past days + ~5 future days."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "past_days": 15, "forecast_days": 6,
        "timezone": "auto",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=timeout)
    r.raise_for_status()
    js = r.json()
    h = pd.DataFrame(js["hourly"])
    h = h.dropna(subset=["temperature_2m", "relative_humidity_2m"])
    daily = hourly_to_daily(h)
    # "today" in the location's own timezone
    tz_offset = pd.Timedelta(seconds=js.get("utc_offset_seconds", 0))
    today = (pd.Timestamp.now(tz="UTC").tz_localize(None) + tz_offset).normalize()
    return daily, today


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"


def geocode(name: str, count: int = 6, timeout: int = 10) -> list[dict]:
    """Place name -> candidate locations (free Open-Meteo geocoding, no key)."""
    r = requests.get(GEOCODE_URL, params={"name": name, "count": count, "language": "en", "format": "json"},
                     timeout=timeout)
    r.raise_for_status()
    return r.json().get("results", []) or []

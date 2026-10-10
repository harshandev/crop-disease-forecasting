"""Download historical hourly weather (free Open-Meteo archive, no API key) and cache as CSV.

Run:  python -m src.fetch_data
"""
import time

import pandas as pd
import requests

from .config import DATA_RAW, END_DATE, HOURLY_VARS, LOCATIONS, START_DATE

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _fetch_year(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "auto",
    }
    for attempt in range(6):
        r = requests.get(ARCHIVE_URL, params=params, timeout=60)
        if r.status_code == 200:
            h = r.json()["hourly"]
            df = pd.DataFrame(h)
            df["time"] = pd.to_datetime(df["time"])
            return df
        wait = 15 * (attempt + 1)
        print(f"   HTTP {r.status_code} ({r.text[:80]!r}); retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {start}..{end} for {lat},{lon}")


def fetch_all() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    years = range(int(START_DATE[:4]), int(END_DATE[:4]) + 1)
    for name, loc in LOCATIONS.items():
        slug = name.split(" ")[0].lower()
        out = DATA_RAW / f"{slug}_hourly.csv"
        if out.exists():
            print(f"skip {name} (already downloaded)")
            continue
        print(f"Fetching {name} ...")
        parts = []
        for y in years:
            parts.append(_fetch_year(loc["lat"], loc["lon"], f"{y}-01-01", f"{y}-12-31"))
            time.sleep(1.0)
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(out, index=False)
        print(f"   saved {len(df):,} rows -> {out.name}")


if __name__ == "__main__":
    fetch_all()

"""Clean the raw hourly files, build daily features and the 5-day-ahead labels.

Run:  python -m src.build_dataset
Output: data/processed/daily.csv  (all locations, one row per day)
"""
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, LOCATIONS
from .features import hourly_to_daily, make_windows


def slug(name: str) -> str:
    return name.split(" ")[0].lower()


def load_daily_all() -> pd.DataFrame:
    parts = []
    for name in LOCATIONS:
        h = pd.read_csv(DATA_RAW / f"{slug(name)}_hourly.csv")
        d = hourly_to_daily(h)
        d.insert(0, "location", name)
        parts.append(d.reset_index())
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    daily = load_daily_all()
    daily.to_csv(DATA_PROCESSED / "daily.csv", index=False, float_format="%.3f")
    print(f"daily.csv: {len(daily):,} rows, {daily['location'].nunique()} locations")

    print("\nShare of days that are 'blight-favourable' (Hutton day) and 5-day-ahead positive rate:")
    for name in LOCATIONS:
        d = daily[daily.location == name].set_index("date").drop(columns="location")
        d.index = pd.to_datetime(d.index)
        _, y, _ = make_windows(d)
        print(f"  {name:28s} hutton days {d['hutton_day'].mean():5.1%}   5-day risk positives {y.mean():5.1%}")


if __name__ == "__main__":
    main()

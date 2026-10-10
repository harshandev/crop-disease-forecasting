"""Create the small sample files users can upload to try the app.  Run: python -m src.make_samples"""
import pandas as pd

from .config import DATA_PROCESSED, DATA_RAW, HOURLY_VARS
from .features import hutton_period_end

OUT = DATA_PROCESSED.parent / "samples"


def main():
    OUT.mkdir(exist_ok=True)
    # 1) hourly weather, Ooty, 2 months (no disease column)
    h = pd.read_csv(DATA_RAW / "ooty_hourly.csv", parse_dates=["time"])
    h = h[(h.time >= "2024-07-15") & (h.time < "2024-09-15")]
    h[["time"] + HOURLY_VARS].round(2).to_csv(OUT / "sample_hourly_weather_ooty.csv", index=False)

    daily = pd.read_csv(DATA_PROCESSED / "daily.csv", parse_dates=["date"])
    nas = daily[daily.location.str.startswith("Nashik")].set_index("date")

    # 2) simple daily weather (what a farm weather station gives), Nashik monsoon 2024
    d = nas.loc["2024-07-01":"2024-09-30"]
    pd.DataFrame({"date": d.index.date, "temp_min": d.t_min.round(1), "temp_max": d.t_max.round(1),
                  "humidity": d.rh_mean.round(0), "rainfall": d.rain_sum.round(1)}).to_csv(
        OUT / "sample_daily_weather_nashik.csv", index=False)

    # 3) labelled daily file for the 'train on my data' demo. DEMO labels come from the weather rule.
    d = nas.loc["2022-01-01":"2024-12-31"]
    lab = hutton_period_end(d).astype(int)
    pd.DataFrame({"date": d.index.date, "temp_min": d.t_min.round(1), "temp_max": d.t_max.round(1),
                  "humidity": d.rh_mean.round(0), "rainfall": d.rain_sum.round(1),
                  "blight_observed": lab.values}).to_csv(OUT / "demo_labelled_daily_nashik_2022_2024.csv", index=False)
    print("blight_observed days:", int(lab.sum()), "of", len(lab))


if __name__ == "__main__":
    main()

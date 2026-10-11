# AI-Driven Early Crop Disease Forecasting — Tomato Late Blight

A web app that estimates, **before any symptoms appear**, the chance that weather over the next
**5 days** will favour Tomato Late Blight — and tells a farmer what to do about it.

**Using the app (no installation):** just open the link you were given in any browser
(laptop or phone). If it says "This app has gone to sleep", press **"Yes, get this app back up!"**
and wait ~30–60 seconds. See `DEMO_GUIDE.md` for a 5-minute walkthrough.

---
## What it does (in plain words)
- **Live forecast for ANY location**: search a place, use your device's GPS, or enter coordinates (not stored).
- **Upload your own weather file (CSV)** → instant risk for every date + the next 5 days, downloadable results.
- **Train models live** in the browser (bundled data or your own labelled file) and see test accuracy, ROC curves and SHAP.
- Looks at the **last 14 days of weather** (temperature, humidity, rain, wind, cloud, sunshine).
- Predicts the **risk (Low / Moderate / High)** of blight-friendly weather in the **next 5 days**.
- Shows **why** (which weather factors raised or lowered the risk) using Explainable AI (SHAP).
- Gives **simple preventive advice** for each risk level.

## Data & models
| Item | Detail |
|---|---|
| Weather data | 10 years (2015–2024), hourly, 6 tomato regions, free Open-Meteo archive |
| Regions | Ooty, Shimla, Darjeeling, Kolar, Nashik, Udaipur |
| Target | "Blight-friendly weather in the next 5 days" from a published agronomic rule (Smith-period style) |
| Models | Logistic Regression, Random Forest, LSTM (deep learning) — trained from scratch, no images, no pre-trained models |
| Explainability | SHAP |
| Split | Train 2015–2021 · tune 2022 · **test 2023–2024 (never seen in training)** |

### Test results (2023–2024)
| Model | ROC-AUC | F1 | Recall |
|---|---|---|---|
| Baseline rule of thumb | 0.920 | 0.757 | 0.781 |
| Logistic Regression | 0.947 | 0.805 | 0.823 |
| Random Forest | 0.953 | 0.807 | 0.830 |
| **LSTM** | **0.954** | 0.804 | **0.877** |

## Honest limitations
- The target is a **weather-based agronomic rule**, not field-recorded outbreaks, so this forecasts
  *disease-favourable weather risk* (the usual basis of blight early-warning systems). It does not
  confirm infection. Validation against real outbreak records is the next step.
- Weather only: no variety, spray history or spore-presence data.
- Live mode uses forecast-model weather, slightly different from the training archive.

## Project layout (for developers)
```
streamlit_app.py        the web app
src/config.py           locations, dates, rule thresholds
src/fetch_data.py       download hourly weather       -> data/raw (not in git)
src/build_dataset.py    daily features + labels       -> data/processed/daily.csv
src/features.py         feature engineering, windows, labels
src/train.py            train + compare all models    -> models/
src/models.py           model code (LSTM runs in plain NumPy — no PyTorch needed to serve)
src/lite.py             upload parsing + lite (daily-data) model
src/make_samples.py     builds the sample upload files -> data/samples/
src/live.py             live weather + place search for the app
components/geolocate/   tiny browser-GPS button component
DEPLOY.md               free hosting steps
```
Rebuild everything: `pip install -r requirements.txt torch matplotlib` then
`python -m src.fetch_data && python -m src.build_dataset && python -m src.train`

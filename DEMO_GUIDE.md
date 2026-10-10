# Demo guide (6 minutes) — for the presenter

**Before the demo:** open the app link ~5 minutes early so it is awake. Internet is preferred
(without it the app still works using saved data, except "Live forecast").

## One-sentence pitch
"Farmers usually notice blight after the leaves are damaged. This app warns them *days earlier*,
from weather alone, so they can scout and protect their crop in time."

## Walkthrough (left menu)
### 1. Upload & predict  ← the main demo
1. Open **"What should the file look like?"** → click **⬇ Sample: daily weather (Nashik)** to download a sample CSV
   (or have your own weather file ready).
2. Choose **Upload my file** and drop the CSV in. The app checks it and says what it found
   ("Daily file detected → 92 days").
3. Result appears instantly: **risk gauge (Low/Moderate/High)**, the next-5-day window, advice.
4. **Why this risk?** → red bars = what pushed risk up (Explainable AI / SHAP).
5. **Risk over time** → forecast for *every* date in the file + **Download results (CSV)**.
6. Try the hourly sample (Ooty) too — it switches to the more precise "full model".
7. Try **"daily + blight records"** sample → **Check vs real records** tab shows how well the forecasts matched the
   recorded blight (with real field records this is the proof of accuracy).

### 2. Train models (live)  ← show the training
1. Click **▶ Train models now**. Watch the steps tick: data split by time → Logistic Regression → Random Forest → SHAP.
2. Results on **unseen test data**: table, ROC curves, alerts-vs-reality grid, "what the model learned".
3. Change the sliders (trees, depth) or switch to **Lite features** and train again.
4. **My own labelled file** → tick the demo file (or upload a daily CSV with `blight_observed`) → **Train on my data** →
   a model is trained on that data and gives a forecast.
5. Open the **LSTM** panel: its training curve (it was trained offline because it needs PyTorch; the weights run here).

### 3. Quick extras
- **Demo scenarios** — real 2023-24 weather: High (Darjeeling), Moderate (Nashik), Low (Udaipur), with "what actually happened".
- **Live forecast (today)** — real weather for a chosen location, right now.
- **Replay a past date** — pick any date in 2023-24.

## Likely questions
- **"Is this detecting disease on plants?"** No — it forecasts *risk from weather*, before symptoms. No photos are used.
- **"How accurate is it?"** On unseen 2023–2024 forecasts ROC-AUC ≈ 0.95 (lite daily model ≈ 0.95 too); it catches ~83–88% of blight-friendly spells.
- **"Which model is best?"** All three beat a simple rule of thumb. LSTM catches the most; Random Forest is equally accurate and easiest to explain.
- **"Trained on real outbreaks?"** The built-in model uses an established agronomic blight rule on 10 years of real weather.
  The Upload + Train pages let anyone test/train with their **own** outbreak records — that is the next validation step.
- **"Which file formats?"** CSV, daily or hourly; column names are matched flexibly.
- **"Does it cost anything?"** No — free weather data and free hosting for the prototype.

## If something goes wrong
- *"App has gone to sleep"* → press **Get this app back up**, wait ~1 minute.
- *"Live weather unavailable"* → no internet; use Upload / Train / Demo scenarios instead.
- *Upload error message* → it says exactly which column is missing; fix the header and re-upload.

# Demo guide (5 minutes) — for the presenter

**Before the demo:** open the app link ~5 minutes early so it is awake. Have internet if possible
(if not, the app falls back to saved sample data and still works).

## The one-sentence pitch
"Farmers usually notice blight after the leaves are damaged. This app warns them *days earlier*,
from weather alone, so they can scout and protect their crop in time."

## Walkthrough
1. **Demo scenarios** (left menu → *Demo scenarios*). These are real weather from 2023–2024.
   - *Humid misty week – Darjeeling*: gauge ~95% → **HIGH**. Point at the blue box: "and blight-friendly
     conditions really did occur in those 5 days."
   - *Building up – Nashik monsoon*: ~55% → **MODERATE**.
   - *Dry spell – Udaipur*: ~2% → **LOW**.
2. Click **Why this risk?** — "Red bars are what pushed the risk up, e.g. a run of humid, mild days."
   (This is the Explainable-AI part.)
3. Click **Weather** — the shaded part is the 5-day forecast window; dotted red line = 11 humid hours/day.
4. Click **What to do** — preventive advice changes with the risk level.
5. **Live forecast (today)** — pick a location; the app pulls today's real weather and gives today's risk.
6. **Model comparison** tab — Logistic Regression, Random Forest and LSTM compared on unseen 2023–2024 data.

## Likely questions
- **"Is this detecting disease on plants?"** No — it forecasts *risk from weather*, before symptoms. No photos are used.
- **"How accurate is it?"** On unseen 2023–2024 forecasts: ROC-AUC ≈ 0.95; it catches ~83–88% of blight-friendly spells.
- **"Which model is best?"** All three learned models beat a simple rule of thumb. LSTM catches the most; Random Forest is equally accurate and easiest to explain.
- **"Was it trained on real outbreaks?"** It uses an established agronomic blight-infection rule on 10 years of real weather.
  Validating with field outbreak records is the next step — we say so openly.
- **"Will it work for my village?"** Pick the nearest region; adding a location is a small change (coordinates only).
- **"Does it cost anything?"** No — free weather data and free hosting for the prototype.

## If something goes wrong
- *"App has gone to sleep"* → press the **Get this app back up** button, wait ~1 minute.
- *Yellow "Live weather unavailable" message* → no internet; app shows saved sample data. Use **Demo scenarios**.

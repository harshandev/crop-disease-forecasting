"""Tomato Late Blight - 5-day early-warning dashboard (Streamlit).

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import DATA_PROCESSED, HORIZON_DAYS, LOCATIONS, WINDOW_DAYS
from src.features import (PRETTY, add_season, make_windows, tabular_feature_names,
                          window_to_tabular)
from src.live import fetch_recent_and_forecast
from src.models import (NumpyLSTM, build_all_windows, fit_sklearn_models, load_metrics,
                        split_masks)

st.set_page_config(page_title="Tomato Blight Early Warning", page_icon="🍅", layout="wide")

LOW, HIGH = 0.30, 0.60
COLORS = {"Low": "#2e9d5b", "Moderate": "#e0a100", "High": "#d93a3a"}
MODEL_NAMES = ["Random Forest", "LSTM (deep learning)", "Logistic Regression"]
FEATURE_NAMES = tabular_feature_names()


# ------------------------------------------------------------------ cached loading
@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "daily.csv", parse_dates=["date"])


@st.cache_resource(show_spinner="Starting up the forecasting engine (first load only)...")
def load_engine():
    daily = load_daily()
    X, y, dates, locs = build_all_windows(daily)
    tr, va, te = split_masks(dates)
    lr, rf = fit_sklearn_models(X[tr], y[tr])
    lstm = NumpyLSTM()
    Xt = window_to_tabular(X)
    probs = {"Logistic Regression": lr.predict_proba(Xt)[:, 1],
             "Random Forest": rf.predict_proba(Xt)[:, 1],
             "LSTM (deep learning)": lstm.predict_proba(X)}
    index = pd.DataFrame({"date": dates, "location": locs, "y": y, "is_test": te})
    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
    except Exception:
        pass
    return {"lr": lr, "rf": rf, "lstm": lstm, "probs": probs, "index": index,
            "explainer": explainer, "X": X}


def predict(engine, X1: np.ndarray, model: str) -> float:
    if model == "LSTM (deep learning)":
        return float(engine["lstm"].predict_proba(X1)[0])
    m = engine["rf"] if model == "Random Forest" else engine["lr"]
    return float(m.predict_proba(window_to_tabular(X1))[0, 1])


def explain(engine, X1: np.ndarray) -> pd.Series | None:
    """SHAP values for one window, grouped into farmer-readable factors."""
    if engine["explainer"] is None:
        return None
    sv = engine["explainer"].shap_values(window_to_tabular(X1))
    sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    s = pd.Series(sv[0], index=FEATURE_NAMES)

    def group(name: str) -> str:
        base = name.split("__")[0]
        if base.startswith("hutton"):
            return "Recent blight-friendly days"
        if base.startswith("doy"):
            return "Time of year"
        return PRETTY.get(base, base)

    return s.groupby(group).sum().sort_values(key=np.abs, ascending=False)


def level_of(p: float) -> str:
    return "Low" if p < LOW else ("Moderate" if p < HIGH else "High")


# ------------------------------------------------------------------ charts
def gauge(p: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p * 100),
        number={"suffix": "%", "font": {"size": 54}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[level_of(p)], "thickness": 0.35},
               "steps": [{"range": [0, LOW * 100], "color": "#e3f4ea"},
                         {"range": [LOW * 100, HIGH * 100], "color": "#fbf1d2"},
                         {"range": [HIGH * 100, 100], "color": "#f9dcdc"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def weather_chart(ctx: pd.DataFrame, forecast_start: pd.Timestamp | None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Temperature (°C)", "Hours of very high humidity (≥90%) and rain (mm)"))
    fig.add_trace(go.Scatter(x=ctx.index, y=ctx["t_max"], name="Day max", line=dict(color="#e4572e")), 1, 1)
    fig.add_trace(go.Scatter(x=ctx.index, y=ctx["t_min"], name="Night min", line=dict(color="#2e86ab")), 1, 1)
    fig.add_trace(go.Bar(x=ctx.index, y=ctx["hrs_rh90"], name="Humid hours (≥90% RH)", marker_color="#6a9bd1"), 2, 1)
    fig.add_trace(go.Bar(x=ctx.index, y=ctx["rain_sum"], name="Rain (mm)", marker_color="#1b4965"), 2, 1)
    fig.add_hline(y=11, line_dash="dot", line_color="#d93a3a", row=2, col=1,
                  annotation_text="11 h = blight-friendly", annotation_position="top left")
    if forecast_start is not None:
        for r in (1, 2):
            fig.add_vrect(x0=forecast_start - pd.Timedelta(hours=12), x1=ctx.index.max() + pd.Timedelta(hours=12),
                          fillcolor="#f2e8cf", opacity=0.45, line_width=0, row=r, col=1)
    fig.update_layout(height=470, barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def shap_chart(contrib: pd.Series) -> go.Figure:
    top = contrib.head(6)[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=["#d93a3a" if v > 0 else "#2e9d5b" for v in top.values]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="← lowers risk      raises risk →")
    return fig


ADVICE = {
    "Low": ["Routine monitoring: walk the field every 3–4 days and look at the lower, older leaves.",
            "Keep plants well spaced and pruned so leaves dry quickly after dew or rain.",
            "No special spray is needed on the basis of weather alone."],
    "Moderate": ["Scout every 1–2 days, especially in low, shaded or poorly ventilated corners of the field.",
                 "Avoid overhead irrigation; irrigate in the morning so foliage dries before night.",
                 "Remove and destroy any leaves with suspicious dark, water-soaked patches.",
                 "Get protective-spray materials ready, so you can act quickly if risk climbs."],
    "High": ["Scout daily, first thing in the morning. Check under leaves for pale-green/dark water-soaked spots.",
             "Consider a protective (preventive) fungicide spray before the humid spell, following your local "
             "agriculture officer / KVK recommendation and the product label.",
             "Do not irrigate overhead; improve airflow by pruning lower leaves and staking.",
             "Remove and bag infected plant parts; do not leave them in the field.",
             "Tell neighbouring growers: blight spreads on wind and rain."],
}


# ------------------------------------------------------------------ sidebar
engine = load_engine()
daily_all = load_daily()
idx = engine["index"]

st.sidebar.title("🍅 Blight Early Warning")
mode = st.sidebar.radio("What do you want to see?",
                        ["Live forecast (today)", "Demo scenarios", "Replay a past date"])
model = st.sidebar.selectbox("Prediction model", MODEL_NAMES,
                             help="Random Forest is the default: it is accurate and explainable.")

ctx_daily, forecast_start, actual, source_note, loc, X1, as_of = None, None, None, "", None, None, None

if mode == "Live forecast (today)":
    loc = st.sidebar.selectbox("Farm location", list(LOCATIONS))
    st.sidebar.caption(LOCATIONS[loc]["note"])
    try:
        d_live, today = fetch_recent_and_forecast(LOCATIONS[loc]["lat"], LOCATIONS[loc]["lon"])
        past = d_live[d_live.index <= today].tail(WINDOW_DAYS)
        if len(past) < WINDOW_DAYS:
            raise ValueError("not enough days returned")
        X1 = make_windows(past, with_labels=False)[0][-1:]
        as_of = past.index[-1]
        ctx_daily = d_live[(d_live.index > as_of - pd.Timedelta(days=WINDOW_DAYS)) &
                           (d_live.index <= as_of + pd.Timedelta(days=HORIZON_DAYS))]
        forecast_start = as_of + pd.Timedelta(days=1)
        source_note = "🟢 Live weather from Open-Meteo (last 14 days + weather outlook)."
    except Exception as e:  # no internet / API down -> bundled sample so the demo never breaks
        st.sidebar.warning("Live weather unavailable - showing the bundled sample (Dec 2024).")
        sub = idx[(idx.location == loc)].iloc[-1]
        as_of = sub.date
        sel = (idx.location == loc) & (idx.date == as_of)
        X1 = engine["X"][np.where(sel)[0][:1]]
        d_loc = daily_all[daily_all.location == loc].set_index("date")
        ctx_daily = d_loc[(d_loc.index > as_of - pd.Timedelta(days=WINDOW_DAYS)) & (d_loc.index <= as_of)]
        source_note = f"🟠 Offline sample data (last bundled day: {as_of:%d %b %Y})."

elif mode == "Demo scenarios":
    test = idx[idx.is_test].copy()
    test["p"] = engine["probs"]["Random Forest"][idx.is_test.values]

    def pick(location, months, target, want_y):
        s = test[(test.location == location) & (test.date.dt.month.isin(months)) & (test.y == want_y)]
        return s.iloc[(s.p - target).abs().argmin()]

    SCEN = {
        "Humid misty week - Darjeeling (high risk)": pick("Darjeeling (W. Bengal)", [6, 7, 8], 0.95, 1),
        "Building up - Nashik monsoon (moderate risk)": pick("Nashik (Maharashtra)", [7, 8, 9], 0.55, 1),
        "Dry spell - Udaipur (low risk)": pick("Udaipur (Rajasthan)", [2, 3, 4], 0.02, 0),
    }
    choice = st.sidebar.radio("Scenario", list(SCEN))
    row = SCEN[choice]
    loc, as_of = row.location, row.date
    st.sidebar.caption("Real weather from 2023-2024 that the model never saw during training.")
    mode = "replay"
else:
    loc = st.sidebar.selectbox("Farm location", list(LOCATIONS))
    test_dates = idx[(idx.location == loc) & idx.is_test].date
    pick_date = st.sidebar.date_input("Forecast made on", value=dt.date(2024, 8, 15),
                                      min_value=test_dates.min().date(), max_value=test_dates.max().date())
    as_of = pd.Timestamp(pick_date)
    mode = "replay"

if mode == "replay":
    sel = np.where((idx.location == loc) & (idx.date == as_of))[0]
    if len(sel) == 0:
        st.error("No data for that date. Try another.")
        st.stop()
    X1 = engine["X"][sel[:1]]
    actual = int(idx.y.iloc[sel[0]])
    d_loc = daily_all[daily_all.location == loc].set_index("date")
    ctx_daily = d_loc[(d_loc.index > as_of - pd.Timedelta(days=WINDOW_DAYS)) &
                      (d_loc.index <= as_of + pd.Timedelta(days=HORIZON_DAYS))]
    forecast_start = as_of + pd.Timedelta(days=1)
    source_note = "🕘 Replay of real past weather (the model only saw data up to the forecast date)."

# ------------------------------------------------------------------ main page
p = predict(engine, X1, model)
level = level_of(p)
win_from, win_to = as_of + pd.Timedelta(days=1), as_of + pd.Timedelta(days=HORIZON_DAYS)

st.title("Tomato Late Blight - 5-Day Early Warning")
st.caption(f"{loc}  •  {source_note}")

c1, c2 = st.columns([1, 1.25])
with c1:
    st.plotly_chart(gauge(p), width='stretch')
with c2:
    st.markdown(f"### Risk: <span style='color:{COLORS[level]}'>{level.upper()}</span>", unsafe_allow_html=True)
    st.markdown(f"**Forecast period:** {win_from:%a %d %b} → {win_to:%a %d %b %Y} (next {HORIZON_DAYS} days)")
    st.markdown(f"**Chance of blight-friendly weather in that window:** {p:.0%}  \n"
                f"<small>Forecast made on {as_of:%d %b %Y} using the previous {WINDOW_DAYS} days of weather · model: {model}</small>",
                unsafe_allow_html=True)
    last3 = ctx_daily[ctx_daily.index <= as_of].tail(3)
    m1, m2, m3 = st.columns(3)
    m1.metric("Humid hours (3d)", f"{last3['hrs_rh90'].mean():.0f} h")
    m2.metric("Avg temp (3d)", f"{last3['t_mean'].mean():.1f} °C")
    m3.metric("Rain (7d)", f"{ctx_daily[ctx_daily.index <= as_of].tail(7)['rain_sum'].sum():.0f} mm")
    if actual is not None:
        happened = "✅ Yes - blight-friendly conditions DID occur" if actual else "❎ No - conditions stayed unfavourable"
        st.info(f"**What actually happened in those 5 days:** {happened}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Why this risk?", "Weather", "What to do", "Model comparison", "About & limits"])

with tab1:
    contrib = explain(engine, X1)
    if contrib is None:
        st.info("Explanation module (SHAP) not available in this environment.")
    else:
        st.markdown("**Which weather factors pushed the risk up (red) or down (green)** "
                    "- computed with SHAP on the Random Forest model.")
        st.plotly_chart(shap_chart(contrib), width='stretch')
        up = contrib[contrib > 0]
        if len(up):
            st.success(f"Biggest reason risk is raised: **{up.index[0]}**.")
        down = contrib[contrib < 0]
        down = down[down < -0.01]
        if len(down):
            st.caption(f"Biggest reassuring factor: {down.idxmin()}.")

with tab2:
    st.plotly_chart(weather_chart(ctx_daily, forecast_start), width='stretch')
    st.caption("Shaded area = the 5-day forecast window. Blight needs long humid spells (≥11 h/day at ≥90% humidity) "
               "with mild temperatures (about 10–27 °C).")

with tab3:
    st.markdown(f"### Suggested actions - {level} risk")
    for a in ADVICE[level]:
        st.markdown(f"- {a}")
    st.caption("General guidance only. Always confirm spray choices and timing with your local agriculture "
               "officer / Krishi Vigyan Kendra.")

with tab4:
    mets = load_metrics()
    df = pd.DataFrame(mets["results"])
    show = df[["model", "roc_auc", "pr_auc", "f1", "precision", "recall"]].rename(columns={
        "model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
        "precision": "Precision", "recall": "Recall"})
    st.markdown(f"Tested on **{mets['n_windows']['test']:,} forecasts from 2023-2024** that no model saw during training "
                f"(trained on 2015-2021, tuned on 2022).")
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]}), hide_index=True, width='stretch')
    fig = go.Figure()
    for r in mets["results"]:
        fig.add_trace(go.Scatter(x=r["roc_curve"]["fpr"], y=r["roc_curve"]["tpr"], mode="lines",
                                 name=f"{r['model']} (AUC {r['roc_auc']:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), name="Random guess"))
    fig.update_layout(height=380, xaxis_title="False alarm rate", yaxis_title="Outbreaks caught (recall)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')
    st.caption("All three learned models beat the simple rule-of-thumb baseline. The LSTM catches the most events "
               "(highest recall); Random Forest is just as accurate overall and easier to explain.")

with tab5:
    st.markdown(f"""
**How it works**
1. Past hourly weather (temperature, humidity, dew point, rain, wind, cloud, sunshine) for 6 tomato regions, 2015-2024, from the free Open-Meteo service.
2. Cleaned and summarised per day (e.g. *hours of ≥90% humidity*, *night temperature*).
3. The model looks at the last **{WINDOW_DAYS} days** and estimates the chance of blight-friendly conditions during the next **{HORIZON_DAYS} days**.
4. Models: Logistic Regression, Random Forest and an LSTM neural network - all trained from scratch (no images, no pre-trained models).
5. SHAP shows which weather factors drove each prediction.

**Important limits (please read)**
- The "blight-friendly" target is built from a published **agronomic rule** (Smith-period style: ≥11 h at ≥90% humidity with night temp ≥10 °C and day temp ≤27 °C on two days in a row), not from field-recorded outbreaks. So the app forecasts *disease-favourable weather risk*, which is the usual basis of blight early-warning systems - it does **not** confirm that plants are infected.
- Not yet validated against real field outbreak records. This is a prototype for demonstration.
- Weather-only: it does not know your variety, spraying history, or whether blight spores are already present.
- Live mode uses forecast-model weather, which differs slightly from the historical data used for training.
- Wake-up note: if the app was idle, the first load can take up to a minute.
""")

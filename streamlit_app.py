"""Tomato Late Blight - 5-day early-warning app (Streamlit).

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import datetime as dt
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src.config import DATA_PROCESSED, HORIZON_DAYS, LOCATIONS, ROOT, WINDOW_DAYS
from src.features import PRETTY, make_windows, tabular_feature_names, window_to_tabular
from src.lite import (LITE_IDX, LITE_PRETTY, group_name, lite_tabular, lite_tabular_names,
                      lite_windows, parse_upload)
from src.live import fetch_recent_and_forecast, geocode
from src.models import (NumpyLSTM, build_all_windows, load_metrics, make_forest, make_logreg,
                        split_masks)

st.set_page_config(page_title="Tomato Blight Early Warning", page_icon="🍅", layout="wide")

SAMPLES = ROOT / "data" / "samples"
_geolocate = components.declare_component("geolocate", path=str(ROOT / "components" / "geolocate"))
LOW, HIGH = 0.30, 0.60
COLORS = {"Low": "#2e9d5b", "Moderate": "#e0a100", "High": "#d93a3a"}
FULL_MODELS = ["Random Forest", "LSTM (deep learning)", "Logistic Regression"]
LITE_MODELS = ["Random Forest", "Logistic Regression"]
FULL_NAMES, LITE_NAMES = tabular_feature_names(), lite_tabular_names()
SAMPLE_FILES = {
    "Use sample: daily weather (Nashik)": "sample_daily_weather_nashik.csv",
    "Use sample: hourly weather (Ooty)": "sample_hourly_weather_ooty.csv",
    "Use sample: daily + blight records": "demo_labelled_daily_nashik_2022_2024.csv",
}


# ================================================================== loading / models
@st.cache_data(show_spinner=False)
def load_daily() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "daily.csv", parse_dates=["date"])


def _tree_explainer(rf):
    try:
        import shap
        return shap.TreeExplainer(rf)
    except Exception:
        return None


@st.cache_resource(show_spinner="Starting up the forecasting engine (first load only)...")
def load_engine():
    daily = load_daily()
    X, y, dates, locs = build_all_windows(daily)
    tr, va, te = split_masks(dates)
    Xt = window_to_tabular(X)
    lr = make_logreg().fit(Xt[tr], y[tr])
    rf = make_forest().fit(Xt[tr], y[tr])
    lstm = NumpyLSTM()
    # lite models: daily numbers only (what a simple farm weather station gives)
    Xl = lite_tabular(X[:, :, LITE_IDX])
    lr_l = make_logreg().fit(Xl[tr], y[tr])
    rf_l = make_forest().fit(Xl[tr], y[tr])
    return {
        "X": X, "index": pd.DataFrame({"date": dates, "location": locs, "y": y, "is_test": te}),
        "full": {"Logistic Regression": lr, "Random Forest": rf, "lstm": lstm, "explainer": _tree_explainer(rf)},
        "lite": {"Logistic Regression": lr_l, "Random Forest": rf_l, "explainer": _tree_explainer(rf_l)},
        "probs": {"Logistic Regression": lr.predict_proba(Xt)[:, 1], "Random Forest": rf.predict_proba(Xt)[:, 1],
                  "LSTM (deep learning)": lstm.predict_proba(X)},
    }


def predict_batch(engine, kind: str, model: str, X: np.ndarray) -> np.ndarray:
    if kind == "full":
        if model == "LSTM (deep learning)":
            return engine["full"]["lstm"].predict_proba(X)
        return engine["full"][model].predict_proba(window_to_tabular(X))[:, 1]
    return engine["lite"][model].predict_proba(lite_tabular(X))[:, 1]


def shap_values(explainer, Xtab: np.ndarray) -> np.ndarray:
    sv = explainer.shap_values(Xtab)
    sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def explain_one(engine, kind: str, X1: np.ndarray) -> pd.Series | None:
    exp = engine[kind]["explainer"]
    if exp is None:
        return None
    names, pretty, tab = ((FULL_NAMES, PRETTY, window_to_tabular(X1)) if kind == "full"
                          else (LITE_NAMES, LITE_PRETTY, lite_tabular(X1)))
    s = pd.Series(shap_values(exp, tab)[0], index=names)
    return s.groupby(lambda n: group_name(n, pretty)).sum().sort_values(key=np.abs, ascending=False)


def level_of(p: float) -> str:
    return "Low" if p < LOW else ("Moderate" if p < HIGH else "High")


# ================================================================== charts
def gauge(p: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(p * 100), number={"suffix": "%", "font": {"size": 54}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS[level_of(p)], "thickness": 0.35},
               "steps": [{"range": [0, LOW * 100], "color": "#e3f4ea"},
                         {"range": [LOW * 100, HIGH * 100], "color": "#fbf1d2"},
                         {"range": [HIGH * 100, 100], "color": "#f9dcdc"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def weather_chart(ctx: pd.DataFrame, forecast_start) -> go.Figure:
    humid = "hrs_rh90" in ctx.columns
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(
        "Temperature (°C)",
        "Hours of very high humidity (≥90%) and rain (mm)" if humid else "Average humidity (%) and rain (mm)"))
    fig.add_trace(go.Scatter(x=ctx.index, y=ctx["t_max"], name="Day max", line=dict(color="#e4572e")), 1, 1)
    fig.add_trace(go.Scatter(x=ctx.index, y=ctx["t_min"], name="Night min", line=dict(color="#2e86ab")), 1, 1)
    if humid:
        fig.add_trace(go.Bar(x=ctx.index, y=ctx["hrs_rh90"], name="Humid hours (≥90% RH)", marker_color="#6a9bd1"), 2, 1)
        fig.add_hline(y=11, line_dash="dot", line_color="#d93a3a", row=2, col=1,
                      annotation_text="11 h = blight-friendly", annotation_position="top left")
    else:
        fig.add_trace(go.Bar(x=ctx.index, y=ctx["rh_mean"], name="Avg humidity (%)", marker_color="#6a9bd1"), 2, 1)
    fig.add_trace(go.Bar(x=ctx.index, y=ctx["rain_sum"], name="Rain (mm)", marker_color="#1b4965"), 2, 1)
    if forecast_start is not None:
        for r in (1, 2):
            fig.add_vrect(x0=forecast_start - pd.Timedelta(hours=12), x1=ctx.index.max() + pd.Timedelta(hours=12),
                          fillcolor="#f2e8cf", opacity=0.45, line_width=0, row=r, col=1)
    fig.update_layout(height=470, barmode="group", margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=-0.12))
    return fig


def bar_chart(contrib: pd.Series, n=6, xtitle="← lowers risk      raises risk →") -> go.Figure:
    top = contrib.head(n)[::-1]
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                           marker_color=["#d93a3a" if v > 0 else "#2e9d5b" for v in top.values]))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=xtitle)
    return fig


def roc_figure(curves: dict) -> go.Figure:
    fig = go.Figure()
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), name="Random guess"))
    fig.update_layout(height=380, xaxis_title="False alarm rate", yaxis_title="Outbreaks caught (recall)",
                      margin=dict(l=10, r=10, t=10, b=10))
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


# ================================================================== shared forecast view
def render_forecast(engine, *, kind, model, X1, ctx, as_of, loc_label, source_note,
                    forecast_start=None, actual=None, extras=None):
    p = float(predict_batch(engine, kind, model, X1)[0])
    level = level_of(p)
    win_from, win_to = as_of + pd.Timedelta(days=1), as_of + pd.Timedelta(days=HORIZON_DAYS)
    st.caption(f"{loc_label}  •  {source_note}")
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.plotly_chart(gauge(p), width="stretch")
    with c2:
        st.markdown(f"### Risk: <span style='color:{COLORS[level]}'>{level.upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"**Forecast period:** {win_from:%a %d %b} → {win_to:%a %d %b %Y} (next {HORIZON_DAYS} days)")
        st.markdown(f"**Chance of blight-friendly weather in that window:** {p:.0%}  \n"
                    f"<small>Forecast made on {as_of:%d %b %Y} using the previous {WINDOW_DAYS} days of weather · "
                    f"model: {model}{' (lite, daily data)' if kind == 'lite' else ''}</small>", unsafe_allow_html=True)
        past = ctx[ctx.index <= as_of]
        m1, m2, m3 = st.columns(3)
        if "hrs_rh90" in past:
            m1.metric("Humid hours (3d)", f"{past.tail(3)['hrs_rh90'].mean():.0f} h")
        else:
            m1.metric("Avg humidity (3d)", f"{past.tail(3)['rh_mean'].mean():.0f} %")
        m2.metric("Avg temp (3d)", f"{past.tail(3)['t_mean'].mean():.1f} °C")
        m3.metric("Rain (7d)", f"{past.tail(7)['rain_sum'].sum():.0f} mm")
        if actual is not None:
            happened = "✅ Yes - blight-friendly conditions DID occur" if actual else "❎ No - conditions stayed unfavourable"
            st.info(f"**What actually happened in those 5 days:** {happened}")

    extras = extras or {}
    tabs = st.tabs(["Why this risk?", "Weather", "What to do", *extras.keys()])
    with tabs[0]:
        contrib = explain_one(engine, kind, X1)
        if contrib is None:
            st.info("Explanation module (SHAP) not available in this environment.")
        else:
            st.markdown("**Which weather factors pushed the risk up (red) or down (green)** - computed with SHAP "
                        f"on the Random Forest{' (lite)' if kind == 'lite' else ''} model.")
            st.plotly_chart(bar_chart(contrib), width="stretch")
            up = contrib[contrib > 0.01]
            if len(up):
                st.success(f"Biggest reason risk is raised: **{up.index[0]}**.")
            down = contrib[contrib < -0.01]
            if len(down):
                st.caption(f"Biggest reassuring factor: {down.idxmin()}.")
    with tabs[1]:
        st.plotly_chart(weather_chart(ctx, forecast_start), width="stretch")
        st.caption("Blight needs long humid spells (≥11 h/day at ≥90% humidity) with mild temperatures (≈10–27 °C)."
                   + (" Shaded area = the 5-day forecast window." if forecast_start is not None else ""))
    with tabs[2]:
        st.markdown(f"### Suggested actions - {level} risk")
        for a in ADVICE[level]:
            st.markdown(f"- {a}")
        st.caption("General guidance only. Always confirm spray choices and timing with your local agriculture "
                   "officer / Krishi Vigyan Kendra.")
    for tab, fn in zip(tabs[3:], extras.values()):
        with tab:
            fn()


# ================================================================== PAGE: upload & predict
def _future_label(obs: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    rng = pd.date_range(obs.index.min(), obs.index.max() + pd.Timedelta(days=HORIZON_DAYS))
    arr = obs.reindex(rng).fillna(0).to_numpy()
    pos = {d: i for i, d in enumerate(rng)}
    return np.array([arr[pos[d] + 1: pos[d] + 1 + HORIZON_DAYS].any() for d in dates], dtype=float)


def page_upload(engine):
    st.title("Upload your weather data → get a blight forecast")
    st.markdown("Upload a weather file (CSV) from your farm, weather station or any weather website. "
                "The app checks it, then forecasts blight risk for **every date** and for **the next 5 days**.")
    with st.expander("What should the file look like? (and sample files to try)"):
        st.markdown("""
**Option A - Daily file** (simple, from any farm weather station). One row per day, at least ~3 weeks:

| date | temp_min | temp_max | humidity | rainfall |
|---|---|---|---|---|
| 2024-08-01 | 22.3 | 26.1 | 92 | 11.3 |

**Option B - Hourly file** (more precise): columns `time`, `temperature_2m`, `relative_humidity_2m`
(optional: `dew_point_2m`, `precipitation`, `wind_speed_10m`, `cloud_cover`, `shortwave_radiation`).

Column names are matched flexibly (e.g. `Tmax`, `Rain`, `RH`). Optional: add a `blight_observed` (0/1) column
to a daily file to **check the forecasts against your real outbreak records**.""")
        cols = st.columns(3)
        for col, (label, fname) in zip(cols, [("⬇ Sample: daily weather (Nashik)", SAMPLE_FILES["Use sample: daily weather (Nashik)"]),
                                              ("⬇ Sample: hourly weather (Ooty)", SAMPLE_FILES["Use sample: hourly weather (Ooty)"]),
                                              ("⬇ Sample: daily + blight records", SAMPLE_FILES["Use sample: daily + blight records"])]):
            col.download_button(label, (SAMPLES / fname).read_bytes(), file_name=fname, mime="text/csv")
        st.caption("The 'blight records' sample uses DEMO labels derived from the weather rule - replace with real field records.")

    src = st.radio("Data source", ["Upload my file", *SAMPLE_FILES], horizontal=True)
    if src == "Upload my file":
        f = st.file_uploader("Drop a CSV file here", type=["csv", "txt"])
        if f is None:
            st.info("👆 Upload a file, or pick one of the samples above to see how it works.")
            return
        try:
            raw, fname = pd.read_csv(f, sep=None, engine="python"), f.name
        except Exception as e:
            st.error(f"I couldn't read that file as a table ({e}). Please upload a plain CSV.")
            return
    else:
        fname = SAMPLE_FILES[src]
        raw = pd.read_csv(SAMPLES / fname)

    up = parse_upload(raw)
    if up.error:
        st.error(up.error)
        st.dataframe(raw.head(), width="stretch")
        return
    for n in up.notes:
        st.success(n)
    with st.expander(f"Preview of your data ({len(raw):,} rows in {fname})"):
        st.dataframe(raw.head(15), width="stretch")

    kind = "full" if up.kind == "hourly" else "lite"
    model = st.selectbox("Prediction model", FULL_MODELS if kind == "full" else LITE_MODELS,
                         help="Random Forest is the default: accurate and explainable.")
    d = up.daily
    X, _, dates = make_windows(d, with_labels=False) if kind == "full" else lite_windows(d)
    if len(X) == 0:
        st.error("Not enough consecutive days to forecast (need 14 days of history).")
        return
    probs = predict_batch(engine, kind, model, X)
    res = pd.DataFrame({"forecast_made_on": dates.date, "risk_probability": probs.round(3),
                        "risk_level": [level_of(p) for p in probs],
                        "forecast_from": (dates + pd.Timedelta(days=1)).date,
                        "forecast_to": (dates + pd.Timedelta(days=HORIZON_DAYS)).date})

    def timeline():
        fig = go.Figure()
        for lo, hi, col in ((0, LOW, "#2e9d5b"), (LOW, HIGH, "#e0a100"), (HIGH, 1, "#d93a3a")):
            fig.add_hrect(y0=lo, y1=hi, fillcolor=col, opacity=0.10, line_width=0)
        fig.add_trace(go.Scatter(x=dates, y=probs, mode="lines", name="Risk", line=dict(color="#1b4965", width=2)))
        if up.blight is not None:
            ob = up.blight[up.blight > 0]
            fig.add_trace(go.Scatter(x=ob.index, y=[1.03] * len(ob), mode="markers", name="Blight observed",
                                     marker=dict(symbol="x", color="#d93a3a", size=7)))
        fig.update_layout(height=380, yaxis=dict(range=[0, 1.08], tickformat=".0%", title="Risk for the next 5 days"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(res.iloc[::-1], hide_index=True, width="stretch", height=240)
        st.download_button("⬇ Download all results (CSV)", res.to_csv(index=False).encode(),
                           file_name="blight_risk_results.csv", mime="text/csv")

    extras = {"Risk over time": timeline}

    if up.blight is not None and kind == "lite":
        def check():
            yv = _future_label(up.blight, dates)
            mask = np.array([t + pd.Timedelta(days=HORIZON_DAYS) <= d.index.max() for t in dates])
            yv, pv = yv[mask], probs[mask]
            st.markdown("**How well did the forecasts match your real blight records?**")
            if len(yv) < 20 or yv.min() == yv.max():
                st.warning("Your records need both blight and no-blight periods to compute accuracy.")
                return
            c = st.columns(3)
            c[0].metric("ROC-AUC", f"{roc_auc_score(yv, pv):.3f}", help="0.5 = random guessing, 1.0 = perfect")
            c[1].metric("Forecast windows checked", f"{len(yv):,}")
            c[2].metric("Windows followed by blight", f"{yv.mean():.0%}")
            tbl = pd.DataFrame({"level": [level_of(p) for p in pv], "blight": yv}).groupby("level").agg(
                forecasts=("blight", "size"), blight_followed_pct=("blight", lambda s: round(100 * s.mean(), 1)))
            st.dataframe(tbl.reindex([lv for lv in ("Low", "Moderate", "High") if lv in tbl.index]), width="stretch")
            st.caption("If the model is useful, 'High' rows should show a much higher % than 'Low' rows.")
        extras["Check vs real records"] = check

    last = len(X) - 1
    ctx = d[(d.index > dates[last] - pd.Timedelta(days=WINDOW_DAYS)) & (d.index <= dates[last])]
    render_forecast(engine, kind=kind, model=model, X1=X[last:last + 1], ctx=ctx, as_of=dates[last],
                    loc_label=f"Your data: {fname}", source_note="Latest forecast, from the last day in your file.",
                    extras=extras)


# ================================================================== PAGE: train models
def _fit_eval(X_tab, y, tr, va, te, names, pretty, rf_params, lr_c):
    models, curves, rows = {}, {}, []
    status = st.status("Training…", expanded=True)
    prog = status.progress(0.0)
    t0 = time.time()
    status.write(f"✔ Data split by time: {tr.sum():,} train · {va.sum():,} validation · {te.sum():,} test windows")
    lr = make_logreg().set_params(logisticregression__C=lr_c)
    t = time.time(); lr.fit(X_tab[tr], y[tr])
    status.write(f"✔ Logistic Regression trained ({time.time() - t:.1f}s)"); prog.progress(0.35)
    rf = make_forest().set_params(**rf_params)
    t = time.time(); rf.fit(X_tab[tr], y[tr])
    status.write(f"✔ Random Forest trained ({rf_params['n_estimators']} trees, {time.time() - t:.1f}s)"); prog.progress(0.7)
    grid = np.linspace(0.05, 0.95, 91)
    for name, m in (("Logistic Regression", lr), ("Random Forest", rf)):
        pv, pt = m.predict_proba(X_tab[va])[:, 1], m.predict_proba(X_tab[te])[:, 1]
        thr = grid[int(np.argmax([f1_score(y[va], pv >= g, zero_division=0) for g in grid]))]
        fpr, tpr, _ = roc_curve(y[te], pt)
        auc = roc_auc_score(y[te], pt)
        curves[name] = (fpr, tpr, auc)
        pred = pt >= thr
        rows.append({"Model": name, "ROC-AUC": auc, "PR-AUC": average_precision_score(y[te], pt),
                     "F1": f1_score(y[te], pred), "Precision": precision_score(y[te], pred, zero_division=0),
                     "Recall": recall_score(y[te], pred), "Alert threshold": thr})
        models[name] = (m, pt, thr)
    imp = None
    try:
        import shap
        sample = np.random.RandomState(0).choice(np.where(te)[0], size=min(300, int(te.sum())), replace=False)
        sv = shap_values(shap.TreeExplainer(rf), X_tab[sample])
        imp = (pd.Series(np.abs(sv).mean(0), index=names)
               .groupby(lambda n: group_name(n, pretty)).sum().sort_values(ascending=False))
        status.write("✔ SHAP importance computed")
    except Exception:
        pass
    prog.progress(1.0)
    status.update(label=f"Training complete in {time.time() - t0:.1f}s", state="complete", expanded=False)
    return {"rows": rows, "curves": curves, "models": models, "imp": imp, "y_te": y[te]}


def _show_training_result(res, extra_rows=None, extra_curves=None):
    df = pd.DataFrame(res["rows"] + (extra_rows or []))
    st.markdown("#### Results on held-out test data (never used for training)")
    st.dataframe(df.style.format({c: "{:.3f}" for c in df.columns[1:]}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**ROC curves**")
        st.plotly_chart(roc_figure({**res["curves"], **(extra_curves or {})}), width="stretch")
    with c2:
        _, pt, thr = res["models"]["Random Forest"]
        cm = confusion_matrix(res["y_te"], pt >= thr, labels=[0, 1])
        st.markdown("**Random Forest - alerts vs reality**")
        fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: no alert", "Predicted: alert"],
                                   y=["Actual: no blight weather", "Actual: blight weather"],
                                   text=cm, texttemplate="%{text}", colorscale="Greens", showscale=False))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Alert threshold {thr:.2f}, chosen on validation data (not on the test set).")
    if res["imp"] is not None:
        st.markdown("**What the model learned (SHAP: average impact of each factor)**")
        st.plotly_chart(bar_chart(res["imp"], n=8, xtitle="average impact on predictions"), width="stretch")


def page_train(engine):
    st.title("Train the models - live")
    st.markdown("Watch the pipeline run: **data → features → train → test → explain**. "
                "Train on the bundled 10-year dataset, or on **your own labelled weather file**.")
    source = st.radio("Training data", ["Bundled 10-year dataset (6 regions)",
                                        "My own labelled file (daily weather + blight_observed)"], horizontal=True)

    if source.startswith("Bundled"):
        daily = load_daily()
        c = st.columns(4)
        c[0].metric("Daily records", f"{len(daily):,}")
        c[1].metric("Regions", daily.location.nunique())
        c[2].metric("Years", f"{daily.date.dt.year.min()}–{daily.date.dt.year.max()}")
        c[3].metric("Weather variables", "7 hourly → 14 daily")
        locs = st.multiselect("Regions to train on", list(LOCATIONS), default=list(LOCATIONS))
        fs = st.radio("Feature set", ["Full (uses hours of high humidity — needs hourly data)",
                                      "Lite (daily min/max temp, humidity, rain only)"], horizontal=True)
        with st.expander("Model settings"):
            a, b, c3 = st.columns(3)
            trees = a.slider("Random Forest trees", 20, 300, 150, 10)
            depth = b.slider("Tree depth", 3, 16, 10)
            lr_c = c3.select_slider("Logistic Regression strength (C)", [0.01, 0.1, 0.5, 1, 5], value=0.5)
        if st.button("▶ Train models now", type="primary") and locs:
            idx = engine["index"]
            sel = idx.location.isin(locs).to_numpy()
            X, y = engine["X"][sel], idx.y.to_numpy()[sel]
            tr, va, te = split_masks(pd.DatetimeIndex(idx.date[sel]))
            lite = fs.startswith("Lite")
            Xtab = lite_tabular(X[:, :, LITE_IDX]) if lite else window_to_tabular(X)
            res = _fit_eval(Xtab, y, tr, va, te, LITE_NAMES if lite else FULL_NAMES,
                            LITE_PRETTY if lite else PRETTY, {"n_estimators": trees, "max_depth": depth}, lr_c)
            res["full_all"] = (not lite) and len(locs) == len(LOCATIONS)
            st.session_state["train_res"] = res
        res = st.session_state.get("train_res")
        if res:
            extra_rows, extra_curves = None, None
            if res.get("full_all"):
                lm = [r for r in load_metrics()["results"] if r["model"].startswith("LSTM")][0]
                extra_rows = [{"Model": "LSTM (pre-trained offline)", "ROC-AUC": lm["roc_auc"], "PR-AUC": lm["pr_auc"],
                               "F1": lm["f1"], "Precision": lm["precision"], "Recall": lm["recall"],
                               "Alert threshold": lm["threshold"]}]
                extra_curves = {"LSTM (pre-trained)": (lm["roc_curve"]["fpr"], lm["roc_curve"]["tpr"], lm["roc_auc"])}
            _show_training_result(res, extra_rows, extra_curves)
        with st.expander("🧠 LSTM (deep learning) — trained offline, shown here"):
            h = pd.DataFrame(load_metrics()["lstm_history"])
            st.markdown("The LSTM reads the 14-day weather sequence day by day. Training it needs PyTorch, which the free "
                        "web host doesn't have - so it was trained once on a laptop, and its learned weights run here. "
                        "Training curve from that run:")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=h.epoch, y=h.train_loss, name="Training loss"), secondary_y=False)
            fig.add_trace(go.Scatter(x=h.epoch, y=h.val_auc, name="Validation AUC"), secondary_y=True)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Epoch")
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Stopped early; best validation score at epoch {int(h.val_auc.idxmax()) + 1}.")
        return

    # ------- user's own labelled data
    st.markdown("Upload a **daily** CSV with: `date, temp_min, temp_max, humidity, rainfall, blight_observed` "
                "(1 = blight recorded that day, else 0). At least ~1 year of days works best.")
    if st.checkbox("Use the demo labelled file (Nashik 2022–2024, rule-derived labels)"):
        raw = pd.read_csv(SAMPLES / SAMPLE_FILES["Use sample: daily + blight records"])
        st.warning("Demo labels are derived from the weather rule, only to show the mechanics. "
                   "Use real field records for real results.")
    else:
        f = st.file_uploader("Labelled daily CSV", type=["csv", "txt"])
        if f is None:
            return
        raw = pd.read_csv(f, sep=None, engine="python")
    up = parse_upload(raw)
    if up.error:
        st.error(up.error)
        return
    if up.kind != "daily" or up.blight is None:
        st.error("This needs a DAILY file that includes a `blight_observed` column (0/1).")
        return
    X, y, dates = lite_windows(up.daily, labels=up.blight)
    n = len(y)
    if n < 150 or y.sum() < 15 or (n - y.sum()) < 15:
        st.error(f"Not enough data to train: {n} usable windows, {int(y.sum())} followed by blight. "
                 "Need ≥150 windows with at least 15 blight and 15 non-blight cases.")
        return
    i1, i2 = int(n * 0.60), int(n * 0.75)
    ar = np.arange(n)
    tr, va, te = ar < i1, (ar >= i1) & (ar < i2), ar >= i2
    c = st.columns(4)
    c[0].metric("Days in file", f"{len(up.daily):,}")
    c[1].metric("Training windows", f"{tr.sum():,}")
    c[2].metric("Blight cases", f"{int(y.sum())} ({y.mean():.0%})")
    c[3].metric("Test period starts", f"{dates[te].min():%b %Y}")
    if y[te].min() == y[te].max():
        st.error("The most recent 25% of your data has only one outcome, so I can't test fairly. Add more variety.")
        return
    if st.button("▶ Train on my data", type="primary"):
        res = _fit_eval(lite_tabular(X), y, tr, va, te, LITE_NAMES, LITE_PRETTY, {"n_estimators": 150, "max_depth": 8}, 0.5)
        res["latest"] = (lite_windows(up.daily)[0][-1:], up.daily.index[-1])
        st.session_state["own_res"] = res
    res = st.session_state.get("own_res")
    if res:
        _show_training_result(res)
        X1, as_of = res["latest"]
        p = float(res["models"]["Random Forest"][0].predict_proba(lite_tabular(X1))[0, 1])
        st.markdown("#### Forecast from the model YOU just trained")
        c1, c2 = st.columns([1, 1.5])
        c1.plotly_chart(gauge(p), width="stretch")
        c2.markdown(f"### Risk: <span style='color:{COLORS[level_of(p)]}'>{level_of(p).upper()}</span>", unsafe_allow_html=True)
        c2.markdown(f"Next 5 days after {as_of:%d %b %Y}: **{p:.0%}** chance of blight being reported.")


# ================================================================== location picker
@st.cache_data(ttl=3600, show_spinner=False)
def _geocode(q: str):
    return geocode(q)


def _km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(a)))


def choose_location():
    """Sidebar location picker. Returns dict(label, lat, lon, preset) or None if nothing chosen yet."""
    how = st.sidebar.radio("Where is your farm?", ["Preset region", "Search any place",
                                                   "My current location (GPS)", "Enter coordinates"])
    if how == "Preset region":
        name = st.sidebar.selectbox("Farm location", list(LOCATIONS))
        st.sidebar.caption(LOCATIONS[name]["note"])
        return {"label": name, "lat": LOCATIONS[name]["lat"], "lon": LOCATIONS[name]["lon"], "preset": name}
    if how == "Search any place":
        q = st.sidebar.text_input("Village / town / city", placeholder="e.g. Kolar, Pune, Nairobi")
        if len(q.strip()) < 2:
            st.sidebar.caption("Type at least 2 letters.")
            return None
        try:
            res = _geocode(q.strip())
        except Exception:
            st.sidebar.error("Place search needs internet. Try 'Enter coordinates' or a preset region.")
            return None
        if not res:
            st.sidebar.warning("No place found. Try another spelling.")
            return None
        opts = {f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}".replace(", ,", ","): r for r in res}
        r = opts[st.sidebar.selectbox("Pick the right place", list(opts))]
        return {"label": f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')}".replace(", ,", ","),
                "lat": r["latitude"], "lon": r["longitude"], "preset": None}
    if how == "My current location (GPS)":
        with st.sidebar:
            st.caption("Your browser will ask permission. Your location is used only to fetch weather for that spot "
                       "and is not stored.")
            val = _geolocate(key="gps_button", default=None)
        if val and val.get("lat") is not None:
            st.session_state["gps"] = val
        elif val and val.get("error"):
            st.sidebar.warning(f"Couldn't read your location ({val['error']}). Use 'Search any place' instead.")
        g = st.session_state.get("gps")
        if not g:
            st.sidebar.info("Click the button above, then choose **Allow**.")
            return None
        return {"label": f"Your location ({g['lat']:.3f}, {g['lon']:.3f})", "lat": g["lat"], "lon": g["lon"], "preset": None}
    c1, c2 = st.sidebar.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, 24.585, format="%.4f")
    lon = c2.number_input("Longitude", -180.0, 180.0, 73.712, format="%.4f")
    return {"label": f"Custom point ({lat:.3f}, {lon:.3f})", "lat": lat, "lon": lon, "preset": None}


# ================================================================== PAGE: live / scenarios / replay
def page_history_modes(engine, mode):
    model = st.sidebar.selectbox("Prediction model", FULL_MODELS,
                                 help="Random Forest is the default: it is accurate and explainable.")
    idx, daily_all = engine["index"], load_daily()
    ctx = forecast_start = actual = None
    if mode == "Live forecast (any location)":
        place = choose_location()
        if place is None:
            st.title("Tomato Late Blight - 5-Day Early Warning")
            st.info("👈 Choose a location in the sidebar to get a live forecast.")
            st.stop()
        loc = place["label"]
        near = min(_km(place["lat"], place["lon"], v["lat"], v["lon"]) for v in LOCATIONS.values())
        try:
            d_live, today = fetch_recent_and_forecast(place["lat"], place["lon"])
            past = d_live[d_live.index <= today].tail(WINDOW_DAYS)
            if len(past) < WINDOW_DAYS:
                raise ValueError("not enough days returned")
            past_w = past.copy()
            if place["lat"] < 0:           # southern hemisphere: seasons are flipped
                past_w.index = past.index + pd.Timedelta(days=182)
            X1 = make_windows(past_w, with_labels=False)[0][-1:]
            as_of = past.index[-1]
            ctx = d_live[(d_live.index > as_of - pd.Timedelta(days=WINDOW_DAYS))
                         & (d_live.index <= as_of + pd.Timedelta(days=HORIZON_DAYS))]
            forecast_start = as_of + pd.Timedelta(days=1)
            note = f"🟢 Live weather for ({place['lat']:.2f}, {place['lon']:.2f}) from Open-Meteo."
            if near > 1000:
                note += (f" ⚠ {near:,.0f} km from the regions the model was trained on - treat as indicative.")
        except Exception:
            if place["preset"] is None:
                st.title("Tomato Late Blight - 5-Day Early Warning")
                st.error("Couldn't fetch live weather for that spot (internet or weather service issue). "
                         "Try again, pick a preset region, or use **Upload & predict** with a weather file.")
                st.stop()
            st.sidebar.warning("Live weather unavailable - showing the bundled sample (Dec 2024).")
            as_of = idx[idx.location == loc].iloc[-1].date
            X1 = engine["X"][np.where((idx.location == loc) & (idx.date == as_of))[0][:1]]
            d_loc = daily_all[daily_all.location == loc].set_index("date")
            ctx = d_loc[(d_loc.index > as_of - pd.Timedelta(days=WINDOW_DAYS)) & (d_loc.index <= as_of)]
            note = f"🟠 Offline sample data (last bundled day: {as_of:%d %b %Y})."
    else:
        if mode == "Demo scenarios":
            test = idx[idx.is_test].copy()
            test["p"] = engine["probs"]["Random Forest"][idx.is_test.values]

            def pick(location, months, target, want_y):
                s = test[(test.location == location) & (test.date.dt.month.isin(months)) & (test.y == want_y)]
                return s.iloc[(s.p - target).abs().argmin()]
            scen = {"Humid misty week - Darjeeling (high risk)": pick("Darjeeling (W. Bengal)", [6, 7, 8], 0.95, 1),
                    "Building up - Nashik monsoon (moderate risk)": pick("Nashik (Maharashtra)", [7, 8, 9], 0.55, 1),
                    "Dry spell - Udaipur (low risk)": pick("Udaipur (Rajasthan)", [2, 3, 4], 0.02, 0)}
            row = scen[st.sidebar.radio("Scenario", list(scen))]
            loc, as_of = row.location, row.date
            st.sidebar.caption("Real weather from 2023-2024 that the model never saw during training.")
        else:
            loc = st.sidebar.selectbox("Farm location", list(LOCATIONS))
            td = idx[(idx.location == loc) & idx.is_test].date
            as_of = pd.Timestamp(st.sidebar.date_input("Forecast made on", value=dt.date(2024, 8, 15),
                                                       min_value=td.min().date(), max_value=td.max().date()))
        sel = np.where((idx.location == loc) & (idx.date == as_of))[0]
        if len(sel) == 0:
            st.error("No data for that date. Try another.")
            st.stop()
        X1, actual = engine["X"][sel[:1]], int(idx.y.iloc[sel[0]])
        d_loc = daily_all[daily_all.location == loc].set_index("date")
        ctx = d_loc[(d_loc.index > as_of - pd.Timedelta(days=WINDOW_DAYS))
                    & (d_loc.index <= as_of + pd.Timedelta(days=HORIZON_DAYS))]
        forecast_start = as_of + pd.Timedelta(days=1)
        note = "🕘 Replay of real past weather (the model only saw data up to the forecast date)."

    st.title("Tomato Late Blight - 5-Day Early Warning")

    def compare():
        mets = load_metrics()
        df = pd.DataFrame(mets["results"])
        show = df[["model", "roc_auc", "pr_auc", "f1", "precision", "recall"]].rename(columns={
            "model": "Model", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1",
            "precision": "Precision", "recall": "Recall"})
        st.markdown(f"Tested on **{mets['n_windows']['test']:,} forecasts from 2023-2024** that no model saw during "
                    "training (trained on 2015-2021, tuned on 2022).")
        st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[1:]}), hide_index=True, width="stretch")
        st.plotly_chart(roc_figure({r["model"]: (r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], r["roc_auc"])
                                    for r in mets["results"]}), width="stretch")

    render_forecast(engine, kind="full", model=model, X1=X1, ctx=ctx, as_of=as_of, loc_label=loc, source_note=note,
                    forecast_start=forecast_start, actual=actual, extras={"Model comparison": compare})


# ================================================================== PAGE: about
def page_about():
    st.title("About this system")
    st.markdown(f"""
**How it works**
1. Past hourly weather (temperature, humidity, dew point, rain, wind, cloud, sunshine) for 6 tomato regions, 2015-2024, from the free Open-Meteo service.
2. Cleaned and summarised per day (e.g. *hours of ≥90% humidity*, *night temperature*).
3. The model looks at the last **{WINDOW_DAYS} days** and estimates the chance of blight-friendly conditions during the next **{HORIZON_DAYS} days**.
4. Models: Logistic Regression, Random Forest and an LSTM neural network - all trained from scratch (no images, no pre-trained models).
5. SHAP shows which weather factors drove each prediction.
6. A **lite** version works from plain daily numbers (min/max temperature, humidity, rain) so farmers can use their own station data.

**Important limits (please read)**
- The built-in "blight-friendly" target is built from a published **agronomic rule** (Smith-period style: ≥11 h at ≥90% humidity with night temp ≥10 °C and day temp ≤27 °C on two days in a row), not from field-recorded outbreaks. The app therefore forecasts *disease-favourable weather risk*, the usual basis of blight early-warning systems; it does **not** confirm infection.
- Not yet validated on field outbreak records - upload yours on the *Upload & predict* page to test it.
- Weather-only: it does not know your variety, spraying history, or whether blight spores are present.
- Live mode works for any location on Earth, but the model was trained on 6 Indian tomato regions - places far from these are shown as *indicative*. It uses forecast-model weather, which differs slightly from the historical training data. Your location is used only to fetch weather and is not stored.
- Prototype for demonstration - not a substitute for local agricultural advice.
- If the app was idle, the first load can take up to a minute.
""")


# ================================================================== main
engine = load_engine()
st.sidebar.title("🍅 Blight Early Warning")
mode = st.sidebar.radio("What do you want to do?", [
    "Upload & predict", "Train models (live)", "Live forecast (any location)", "Demo scenarios",
    "Replay a past date", "About & limits"])

if mode == "Upload & predict":
    page_upload(engine)
elif mode == "Train models (live)":
    page_train(engine)
elif mode == "About & limits":
    page_about()
else:
    page_history_modes(engine, mode)

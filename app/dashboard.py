"""A small dashboard to show the results. Run it with:

    pip install -e ".[dashboard]"
    streamlit run app/dashboard.py

It reads the database (data/mandi.duckdb) and the backtest output
(reports/forecasts.parquet), so run `mandi load` and `mandi backtest` first.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mandi import config
from mandi.db import connect, load_daily
from mandi.series import make_panel

st.set_page_config(page_title="Mandi price spike alerts", layout="wide")
st.title("Tomato, onion and potato: price forecasts and spike alerts")


@st.cache_data  # don't re-read the files every time a widget changes
def load():
    con = connect()
    panel = make_panel(load_daily(con, "state"))
    fc = pd.read_parquet(config.REPORTS_DIR / "forecasts.parquet")
    events = pd.read_csv(config.REPORTS_DIR / "spike_events.csv", parse_dates=["start", "end"])
    acc = pd.read_csv(config.REPORTS_DIR / "accuracy.csv")
    alerts = pd.read_csv(config.REPORTS_DIR / "alerts.csv")
    return panel, fc, events, acc, alerts


panel, fc, events, acc, alerts = load()

col1, col2, col3 = st.columns(3)
series = col1.selectbox("Commodity and state", sorted(panel["series_id"].unique()))
model = col2.selectbox("Model", sorted(fc["model"].unique()), index=0)
horizon = col3.slider("Days ahead", 1, config.HORIZON, 1)

hist = panel[panel["series_id"] == series]
f = fc[(fc["series_id"] == series) & (fc["model"] == model) & (fc["horizon"] == horizon) & fc["is_test"]]

fig = go.Figure()
fig.add_trace(go.Scatter(x=f["target_date"], y=f["upper"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=f["target_date"], y=f["lower"], fill="tonexty", line=dict(width=0),
                         name="90% interval", fillcolor="rgba(37,99,235,0.15)"))
fig.add_trace(go.Scatter(x=f["target_date"], y=f["forecast"], name="forecast", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=hist["date"], y=hist["price"], name="actual price", line=dict(color="#111827")))
for _, e in events[events["series_id"] == series].iterrows():
    fig.add_vrect(x0=e["start"], x1=e["end"], fillcolor="rgba(220,38,38,0.12)", line_width=0)
fig.update_layout(yaxis_title="Rs per kg (state median modal price)", height=480,
                  margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, width="stretch")
st.caption("Red shading = spike events (price above 1.5x its 90-day median for 3+ days).")

st.subheader("How accurate is each model?")
st.dataframe(acc[acc["horizon"] == horizon].round(3), hide_index=True)
st.subheader("How good are the spike alerts?")
st.dataframe(alerts.round(3), hide_index=True)

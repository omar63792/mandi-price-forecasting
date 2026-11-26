"""Price spikes: what counts as one, and how good are our alerts?

GROUND TRUTH ("did a spike actually happen?")
A spike day = the price is more than SPIKE_RATIO (1.5x) its own median over the
previous SPIKE_LOOKBACK (90) days. A spike EVENT = a run of at least
SPIKE_MIN_DAYS such days. This rule is simple, written down before looking at
results, and independent of any model, so no model can "define its own success".

HONEST CAVEAT (put this in your README): the rule is a choice. Change 1.5 to
1.3 and you get more, smaller events. Report how results move when you change it,
and cross-check a few events against news (data/known_events.csv).

ALERTS
1. conformal alert: today's price is ABOVE the upper edge of the 90% interval of
   yesterday's 1-day-ahead forecast. In words: "today's price is higher than our
   model thought was plausible". Same idea as an anomaly detector.
2. baseline alert: the price rose more than 20% versus 7 days ago. Any alert
   system must beat this one-line rule to be worth building.

SCORING
* Several alert days in a row are one alert "episode" (a person would get one
  message, not five), so we merge alerts less than 4 days apart.
* recall    = share of spike events that got an alert between 7 days before
              and 3 days after the event started (timely detection)
* precision = share of alert episodes that started inside some event window
              (7 days before its start .. its end)
* false alarms per series per year: the cost a user pays in pointless alerts
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mandi.config import (ALERT_LAG_DAYS, ALERT_LEAD_DAYS, SPIKE_LOOKBACK, SPIKE_MIN_DAYS,
                          SPIKE_RATIO)


def label_spike_events(panel: pd.DataFrame, ratio: float = SPIKE_RATIO,
                       lookback: int = SPIKE_LOOKBACK, min_days: int = SPIKE_MIN_DAYS) -> pd.DataFrame:
    """Return one row per spike event: series_id, start, end, peak_ratio."""
    events = []
    for sid, g in panel.groupby("series_id"):
        price = g.set_index("date")["price"]
        # .shift(1): the reference median must not include today itself
        ref = price.shift(1).rolling(lookback, min_periods=lookback // 2).median()
        is_spike = (price > ratio * ref).fillna(False)
        # label consecutive runs: the run id increases every time is_spike flips
        run_id = (is_spike != is_spike.shift()).cumsum()
        for _, run in is_spike[is_spike].groupby(run_id[is_spike]):
            if len(run) >= min_days:
                start, end = run.index.min(), run.index.max()
                peak = float((price[start:end] / ref[start:end]).max())
                events.append((sid, start, end, peak))
    return pd.DataFrame(events, columns=["series_id", "start", "end", "peak_ratio"])


def conformal_alerts(fc: pd.DataFrame, model: str) -> pd.DataFrame:
    """Alert days where the actual price beat the 1-day-ahead upper bound."""
    d = fc[(fc["model"] == model) & (fc["horizon"] == 1)].dropna(subset=["actual", "upper"])
    d = d[d["actual"] > d["upper"]]
    return d[["series_id", "target_date"]].rename(columns={"target_date": "date"})


def baseline_alerts(panel: pd.DataFrame, rise: float = 0.20) -> pd.DataFrame:
    """Alert days where price is more than `rise` above its value 7 days earlier."""
    out = []
    for sid, g in panel.groupby("series_id"):
        price = g.set_index("date")["price"]
        hit = price > (1 + rise) * price.shift(7)
        out.append(pd.DataFrame({"series_id": sid, "date": price.index[hit.fillna(False).values]}))
    return pd.concat(out, ignore_index=True)


def to_episodes(alerts: pd.DataFrame, gap_days: int = 3) -> pd.DataFrame:
    """Merge alert days that are at most `gap_days` apart into one episode (keep the first day)."""
    eps = []
    for sid, g in alerts.sort_values("date").groupby("series_id"):
        new_episode = g["date"].diff().dt.days.fillna(np.inf) > gap_days
        eps.append(g[new_episode.values])
    return pd.concat(eps, ignore_index=True) if eps else alerts.iloc[0:0]


def score_alerts(alerts: pd.DataFrame, events: pd.DataFrame, period_start, period_end,
                 n_series: int) -> dict:
    """Precision, recall and false-alarm rate inside an evaluation period."""
    lead, lag = pd.Timedelta(days=ALERT_LEAD_DAYS), pd.Timedelta(days=ALERT_LAG_DAYS)
    ev = events[(events["start"] >= period_start) & (events["start"] <= period_end)]
    al = to_episodes(alerts[(alerts["date"] >= period_start) & (alerts["date"] <= period_end)])

    detected = 0
    for _, e in ev.iterrows():
        a = al[(al["series_id"] == e["series_id"]) & al["date"].between(e["start"] - lead, e["start"] + lag)]
        detected += len(a) > 0
    true_alerts = 0
    for _, a in al.iterrows():
        e = events[(events["series_id"] == a["series_id"])
                   & (a["date"] >= events["start"] - lead) & (a["date"] <= events["end"])]
        true_alerts += len(e) > 0

    years = max((pd.Timestamp(period_end) - pd.Timestamp(period_start)).days / 365.25, 1e-9)
    return {
        "events": len(ev),
        "alert_episodes": len(al),
        "recall": detected / len(ev) if len(ev) else np.nan,
        "precision": true_alerts / len(al) if len(al) else np.nan,
        "false_alarms_per_series_year": (len(al) - true_alerts) / (n_series * years),
    }

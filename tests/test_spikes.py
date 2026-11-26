import numpy as np
import pandas as pd

from mandi.spikes import label_spike_events, score_alerts


def panel_with_one_spike():
    dates = pd.date_range("2023-01-01", periods=300, freq="D")
    price = np.full(300, 20.0)
    price[200:215] = 40.0  # a 15-day doubling
    return pd.DataFrame({"series_id": "Tomato | X", "date": dates, "price": price})


def test_spike_is_found():
    ev = label_spike_events(panel_with_one_spike())
    assert len(ev) == 1
    assert ev.loc[0, "start"] == pd.Timestamp("2023-01-01") + pd.Timedelta(days=200)


def test_alert_scoring():
    panel = panel_with_one_spike()
    ev = label_spike_events(panel)
    start = ev.loc[0, "start"]
    alerts = pd.DataFrame({"series_id": ["Tomato | X", "Tomato | X"],
                           "date": [start - pd.Timedelta(days=2), pd.Timestamp("2023-03-01")]})
    s = score_alerts(alerts, ev, panel["date"].min(), panel["date"].max(), n_series=1)
    assert s["recall"] == 1.0      # the spike was caught
    assert s["precision"] == 0.5   # one of two alerts was a false alarm

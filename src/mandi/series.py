"""Turn the SQL view into clean, gap-free daily time series.

A "series" in this project = one commodity in one state (or All India), e.g.
"Tomato | Maharashtra". Each series becomes a row-per-day table with no missing
calendar days, because forecasting models assume equally spaced time steps.
"""

from __future__ import annotations

import pandas as pd

from mandi.config import MAX_FFILL_DAYS, MIN_HISTORY_DAYS


def make_panel(daily: pd.DataFrame, min_history_days: int = MIN_HISTORY_DAYS) -> pd.DataFrame:
    """Return a 'long' panel with columns: series_id, state, commodity, date, price, arrivals, imputed.

    * Missing days (holidays, Sundays, no reports) are forward-filled for up to
      MAX_FFILL_DAYS days. Longer gaps stay missing, and rows with no price are
      dropped from training. We record `imputed` so evaluation can skip them:
      scoring a model against a value WE invented would flatter it.
    """
    out = []
    for (state, commodity), g in daily.groupby(["state", "commodity"]):
        g = g.set_index(pd.to_datetime(g["price_date"])).sort_index()
        full_index = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_index)
        imputed = g["price_rs_kg"].isna()
        price = g["price_rs_kg"].ffill(limit=MAX_FFILL_DAYS)
        s = pd.DataFrame({
            "series_id": f"{commodity} | {state}",
            "state": state,
            "commodity": commodity,
            "date": full_index,
            "price": price.values,
            "arrivals": g["arrivals_tonnes"].values,
            "imputed": imputed.values,
        })
        if s["price"].notna().sum() >= min_history_days:
            out.append(s)
    if not out:
        raise ValueError("No series has enough history. Download more data (see docs/DATA.md).")
    return pd.concat(out, ignore_index=True)

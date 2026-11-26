"""Backtesting: how would each model have done if we had used it in the past?

ROLLING-ORIGIN EVALUATION (the only honest way to test a forecaster)
Pick a past date, pretend it's "today", train only on data up to that day,
forecast the next 7 days, compare with what really happened. Move "today"
forward one day and repeat. Never shuffle time series into random train/test
splits: that lets the model peek at the future.

TIMELINE
    |------ training history ------|-- calibration --|------- test -------|
                                   ^ conformal needs some past errors before
                                     its intervals mean anything, so we start
                                     forecasting `calib_days` before the test
                                     period but only SCORE the test period.

LightGBM is re-trained every `refit_every` days (re-training daily is slow and
barely changes anything); each re-train only sees data up to that day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mandi.config import CONFORMAL_WINDOW
from mandi.models import MODELS


def _actuals(panel: pd.DataFrame) -> pd.DataFrame:
    """Observed prices only. Forward-filled (imputed) days are NOT ground truth."""
    real = panel[~panel["imputed"] & panel["price"].notna()]
    return real[["series_id", "date", "price"]].rename(columns={"date": "target_date", "price": "actual"})


def run_backtest(panel: pd.DataFrame, model_names: list[str], test_days: int = 365,
                 calib_days: int = CONFORMAL_WINDOW, refit_every: int = 90) -> pd.DataFrame:
    last = panel["date"].max()
    test_start = last - pd.Timedelta(days=test_days - 1)
    first_origin = test_start - pd.Timedelta(days=calib_days)
    all_origins = pd.date_range(first_origin, last - pd.Timedelta(days=1), freq="D")

    results = []
    for name in model_names:
        print(f"  backtesting {name} ...")
        blocks = [all_origins[i:i + refit_every] for i in range(0, len(all_origins), refit_every)]
        for block in blocks:
            train = panel[panel["date"] <= block[0]]  # nothing after the block's first day
            model = MODELS[name]().fit(train)
            visible = panel[panel["date"] <= block[-1]]  # features only look backwards
            fc = model.predict(visible, block)
            fc["model"] = name
            results.append(fc)

    fc = pd.concat(results, ignore_index=True)
    fc = fc.merge(_actuals(panel), on=["series_id", "target_date"], how="left")
    fc["is_test"] = fc["origin"] >= test_start
    return fc


def accuracy_table(fc: pd.DataFrame, baseline: str = "naive", n_boot: int = 500,
                   block_days: int = 14, seed: int = 0) -> pd.DataFrame:
    """MAE (Rs/kg) per model and horizon, plus 'relative MAE' vs the naive model
    with a 90% bootstrap interval.

    relative MAE < 1  -> better than naive.  0.9 means 10% smaller errors.

    Bootstrap: resample blocks of 14 consecutive origin days (not single days,
    because errors on neighbouring days are correlated) and recompute the ratio
    500 times. If the interval contains 1.0, we can't claim the model is better.
    """
    d = fc[fc["is_test"]].dropna(subset=["actual", "forecast"]).copy()
    d["abs_err"] = (d["actual"] - d["forecast"]).abs()
    d["ape"] = d["abs_err"] / d["actual"]
    # compare models on exactly the same (series, origin, horizon) rows
    wide = d.pivot_table(index=["series_id", "origin", "horizon"], columns="model", values="abs_err").dropna()

    rng = np.random.default_rng(seed)
    rows = []
    for h, g in wide.groupby(level="horizon"):
        days = g.index.get_level_values("origin")
        day_codes = (days - days.min()).days // block_days
        blocks = np.unique(day_codes)
        for model in g.columns:
            ratio = g[model].mean() / g[baseline].mean()
            boots = []
            for _ in range(n_boot):
                pick = rng.choice(blocks, size=len(blocks), replace=True)
                counts = pd.Series(pick).value_counts()
                w = pd.Series(day_codes).map(counts).fillna(0).values
                boots.append((g[model].values * w).sum() / (g[baseline].values * w).sum())
            lo, hi = np.percentile(boots, [5, 95])
            rows.append({
                "model": model, "horizon": h, "mae_rs_kg": g[model].mean(),
                "mape_pct": 100 * d[(d["model"] == model) & (d["horizon"] == h)]["ape"].mean(),
                "rel_mae_vs_naive": ratio, "rel_mae_ci90_low": lo, "rel_mae_ci90_high": hi,
                "n": len(g),
            })
    return pd.DataFrame(rows).sort_values(["horizon", "rel_mae_vs_naive"])

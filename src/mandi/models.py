"""Forecasting models.

Every model has the same two methods, so backtest.py can treat them alike:

    model.fit(train_panel)               learn from history (some models learn nothing)
    model.predict(history, origins)      for each origin date, forecast days 1..HORIZON ahead

`origins` are the "today" dates on which a forecast is made. The output is a
long table: series_id, origin, horizon, target_date, forecast.

WHY START WITH NAIVE MODELS?
A fancy model is only interesting if it beats a trivial one. "Tomorrow's price =
today's price" is surprisingly hard to beat for daily prices. Every result in
this project is reported RELATIVE to that baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mandi.config import HORIZON


def _empty_forecast_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["series_id", "origin", "horizon", "target_date", "forecast"])


# ---------------------------------------------------------------------------
# 1. Naive: forecast = last observed price, for every horizon
# ---------------------------------------------------------------------------
class Naive:
    name = "naive"

    def fit(self, panel: pd.DataFrame) -> "Naive":
        return self  # nothing to learn

    def predict(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for sid, g in panel.groupby("series_id"):
            price = g.set_index("date")["price"]
            for origin in origins:
                last = price.get(origin, np.nan)  # the price known on the origin day
                for h in range(1, HORIZON + 1):
                    rows.append((sid, origin, h, origin + pd.Timedelta(days=h), last))
        return pd.DataFrame(rows, columns=_empty_forecast_frame().columns)


# ---------------------------------------------------------------------------
# 2. Seasonal naive: forecast = price on the same weekday last week
# ---------------------------------------------------------------------------
class SeasonalNaive:
    """Captures weekly patterns (e.g. lower prices after heavy Monday arrivals)."""

    name = "seasonal_naive_7"

    def __init__(self, season: int = 7):
        self.season = season

    def fit(self, panel: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for sid, g in panel.groupby("series_id"):
            price = g.set_index("date")["price"]
            for origin in origins:
                for h in range(1, HORIZON + 1):
                    # go back whole seasons until we land on a date we already know
                    k = int(np.ceil(h / self.season))
                    src = origin + pd.Timedelta(days=h - k * self.season)
                    rows.append((sid, origin, h, origin + pd.Timedelta(days=h), price.get(src, np.nan)))
        return pd.DataFrame(rows, columns=_empty_forecast_frame().columns)


# ---------------------------------------------------------------------------
# 3. LightGBM: a gradient-boosted tree model on hand-made features
# ---------------------------------------------------------------------------
def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build features that use ONLY information available on each date.

    Using information from the future by accident ("leakage") is the most common
    way forecasting projects fool themselves. Every feature below looks backwards:
    shift(k) moves values k days later, so row t sees the price from t-k.

    We model log prices. Why: price changes are multiplicative (a 20% jump is a
    20% jump whether tomato is Rs 10 or Rs 50), and logs turn ratios into differences.
    """
    out = []
    for sid, g in panel.groupby("series_id"):
        g = g.sort_values("date").copy()
        lp = np.log(g["price"])
        f = pd.DataFrame(index=g.index)
        f["series_id"] = sid
        f["date"] = g["date"]
        f["commodity"] = g["commodity"]
        f["log_price"] = lp
        # recent changes: how much did the price move over the last 1, 3, 7, 14, 28 days?
        for k in [1, 3, 7, 14, 28]:
            f[f"ret_{k}"] = lp - lp.shift(k)
        # where is today's price relative to recent averages? (momentum / mean reversion)
        for w in [7, 28, 90]:
            f[f"dev_ma_{w}"] = lp - lp.rolling(w, min_periods=w // 2).mean()
        f["vol_28"] = lp.diff().rolling(28, min_periods=14).std()  # recent volatility
        # same time last year: seasonal harvest cycles repeat every year
        f["yoy"] = lp - lp.shift(364)
        # arrivals: when less produce reaches the mandi, prices tend to rise
        arr = np.log1p(g["arrivals"])
        f["arr_dev_28"] = arr - arr.rolling(28, min_periods=7).mean()
        # calendar: sin/cos make Dec 31 and Jan 1 "close together"
        doy = g["date"].dt.dayofyear
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        f["dow"] = g["date"].dt.dayofweek
        f["imputed"] = g["imputed"]
        # targets: change in log price from today to h days ahead (one column per h)
        for h in range(1, HORIZON + 1):
            f[f"target_{h}"] = lp.shift(-h) - lp
            f[f"target_imputed_{h}"] = g["imputed"].shift(-h)
        out.append(f)
    feats = pd.concat(out, ignore_index=True)
    feats["commodity"] = feats["commodity"].astype("category")
    return feats


FEATURE_COLUMNS = [
    "commodity", "ret_1", "ret_3", "ret_7", "ret_14", "ret_28",
    "dev_ma_7", "dev_ma_28", "dev_ma_90", "vol_28", "yoy", "arr_dev_28",
    "doy_sin", "doy_cos", "dow",
]


class LightGBMDirect:
    """One LightGBM model per horizon (the 'direct' strategy), shared by all series
    (a 'global' model). Global models learn from every state and commodity at once,
    which matters when each single series is short and noisy."""

    name = "lightgbm"

    def __init__(self, **params):
        self.params = dict(
            n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        )
        self.params.update(params)
        self.models: dict[int, object] = {}

    def fit(self, panel: pd.DataFrame) -> "LightGBMDirect":
        import lightgbm as lgb  # imported here so the naive models work without it

        feats = make_features(panel)
        for h in range(1, HORIZON + 1):
            # train only on rows whose target is a REAL observed price
            ok = feats[f"target_{h}"].notna() & (feats[f"target_imputed_{h}"] == False)  # noqa: E712
            model = lgb.LGBMRegressor(objective="l1", **self.params)  # l1 = predict the median
            model.fit(feats.loc[ok, FEATURE_COLUMNS], feats.loc[ok, f"target_{h}"])
            self.models[h] = model
        return self

    def predict(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        feats = make_features(panel)
        feats = feats[feats["date"].isin(origins) & feats["log_price"].notna()]
        rows = []
        for h, model in self.models.items():
            pred_change = model.predict(feats[FEATURE_COLUMNS])
            rows.append(pd.DataFrame({
                "series_id": feats["series_id"].values,
                "origin": feats["date"].values,
                "horizon": h,
                "target_date": feats["date"].values + np.timedelta64(h, "D"),
                "forecast": np.exp(feats["log_price"].values + pred_change),
            }))
        return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Chronos: a pretrained "foundation model" for time series (optional)
# ---------------------------------------------------------------------------
class ChronosZeroShot:
    """Amazon's Chronos-Bolt, used 'zero-shot': no training on our data at all.
    The question it answers in an interview: does a model pretrained on millions
    of other time series beat one we trained ourselves?

    Install with:  pip install -e ".[chronos]"   (downloads ~200 MB of PyTorch)
    """

    name = "chronos_bolt_small"

    def __init__(self, model_id: str = "amazon/chronos-bolt-small", context_days: int = 512):
        self.model_id = model_id
        self.context_days = context_days
        self.pipeline = None

    def fit(self, panel: pd.DataFrame) -> "ChronosZeroShot":
        from chronos import BaseChronosPipeline

        self.pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map="cpu")
        return self

    def predict(self, panel: pd.DataFrame, origins: pd.DatetimeIndex) -> pd.DataFrame:
        import torch

        rows = []
        for sid, g in panel.groupby("series_id"):
            price = g.set_index("date")["price"]
            for origin in origins:
                ctx = price.loc[:origin].tail(self.context_days)
                if ctx.notna().sum() < 60:
                    continue
                context = torch.tensor(np.log(ctx.values), dtype=torch.float32)  # NaN is allowed
                quantiles, _mean = self.pipeline.predict_quantiles(
                    context=context, prediction_length=HORIZON, quantile_levels=[0.5]
                )
                median = np.exp(quantiles[0, :, 0].numpy())
                for h in range(1, HORIZON + 1):
                    rows.append((sid, origin, h, origin + pd.Timedelta(days=h), float(median[h - 1])))
        return pd.DataFrame(rows, columns=_empty_forecast_frame().columns)


MODELS = {m.name: m for m in [Naive, SeasonalNaive, LightGBMDirect, ChronosZeroShot]}

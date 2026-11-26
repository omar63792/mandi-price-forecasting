"""Conformal prediction: turn a point forecast into an interval you can trust.

THE IDEA IN ONE PARAGRAPH
Look at how wrong the model has been recently. If 90% of its recent errors were
smaller than Rs 4/kg, then "forecast +/- Rs 4" will contain the truth about 90%
of the time, as long as the future behaves like the recent past. No assumption
that errors are normally distributed. That's split conformal prediction.

TWO TIME-SERIES WRINKLES
1. No peeking. A forecast made on day s for day s+h only has a known error once
   day s+h has passed. So the interval for a forecast made today may only use
   errors whose target_date <= today. (`_known_errors` enforces this.)
2. The world changes. A fixed recipe can drift away from 90% coverage when
   volatility rises (e.g. during a spike). Adaptive Conformal Inference (ACI,
   Gibbs & Candes, NeurIPS 2021) nudges the error level after every miss:
       alpha_next = alpha_now + gamma * (alpha_target - missed)
   Miss more often than 10% -> alpha shrinks -> intervals widen, and vice versa.

We work with errors on the LOG scale (log(actual) - log(forecast)), so a Rs 4
error on a Rs 10 tomato is treated as much bigger than on a Rs 60 tomato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mandi.config import ACI_GAMMA, ALPHA, CONFORMAL_WINDOW


def _conformal_quantile(abs_errors: np.ndarray, alpha: float) -> float:
    """The (1-alpha) quantile with the small-sample correction from conformal theory:
    use the ceil((n+1)(1-alpha))-th smallest error instead of the plain quantile."""
    n = len(abs_errors)
    if n == 0:
        return np.nan
    if alpha <= 0:
        return np.inf  # asked for 100% coverage: infinitely wide interval
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(abs_errors)[k - 1])


def add_intervals(
    fc: pd.DataFrame,
    alpha: float = ALPHA,
    window: int = CONFORMAL_WINDOW,
    gamma: float = ACI_GAMMA,
    adaptive: bool = True,
) -> pd.DataFrame:
    """Add `lower`, `upper` columns to a backtest table.

    `fc` must have: series_id, origin, horizon, target_date, forecast, actual.
    Intervals are computed separately for each (series, horizon), because a
    7-day-ahead forecast is naturally less certain than a 1-day-ahead one.
    """
    fc = fc.sort_values(["series_id", "horizon", "origin"]).copy()
    fc["log_err"] = np.log(fc["actual"]) - np.log(fc["forecast"])
    fc["lower"] = np.nan
    fc["upper"] = np.nan
    fc["alpha_used"] = np.nan

    for _, g in fc.groupby(["series_id", "horizon"], sort=False):
        origins = g["origin"].values
        targets = g["target_date"].values
        errs = g["log_err"].values
        preds = g["forecast"].values
        actuals = g["actual"].values
        alpha_t = alpha
        lower, upper, used = [], [], []
        next_to_learn = 0  # index of the oldest forecast whose outcome ACI hasn't used yet
        for i in range(len(g)):
            # --- ACI: learn from every past forecast whose outcome is now known ---
            # (a 7-day-ahead forecast only "resolves" 7 days after it was made)
            if adaptive:
                while next_to_learn < i and targets[next_to_learn] <= origins[i]:
                    j = next_to_learn
                    if not np.isnan(errs[j]) and np.isfinite(upper[j]):
                        missed = float(not (lower[j] <= actuals[j] <= upper[j]))
                        alpha_t = alpha_t + gamma * (alpha - missed)
                    next_to_learn += 1
            # --- split conformal on the errors we are allowed to know today ---
            known = (targets[:i] <= origins[i]) & ~np.isnan(errs[:i])
            recent = np.abs(errs[:i][known])[-window:]
            q = _conformal_quantile(recent, float(np.clip(alpha_t, 0.0, 1.0)))
            lower.append(preds[i] * np.exp(-q))
            upper.append(preds[i] * np.exp(q))
            used.append(alpha_t)
        fc.loc[g.index, "lower"] = lower
        fc.loc[g.index, "upper"] = upper
        fc.loc[g.index, "alpha_used"] = used
    return fc


def coverage_report(fc: pd.DataFrame) -> pd.DataFrame:
    """Did 90% intervals contain the truth 90% of the time? Also: how wide were they?"""
    d = fc.dropna(subset=["actual", "lower", "upper"])
    d = d[np.isfinite(d["upper"])]
    d = d.assign(
        covered=(d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"]),
        width_pct=100 * (d["upper"] - d["lower"]) / d["forecast"],
    )
    return d.groupby(["model", "horizon"]).agg(
        coverage=("covered", "mean"), mean_width_pct=("width_pct", "mean"), n=("covered", "size")
    ).reset_index()

import numpy as np
import pandas as pd

from mandi.conformal import add_intervals


def fake_forecasts(n=800, seed=0):
    """A forecaster whose log-errors are random noise: intervals should cover ~90%."""
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2022-01-01", periods=n, freq="D")
    forecast = np.full(n, 20.0)
    actual = forecast * np.exp(rng.normal(0, 0.1, n))
    return pd.DataFrame({"series_id": "s", "origin": origins, "horizon": 1,
                         "target_date": origins + pd.Timedelta(days=1),
                         "forecast": forecast, "actual": actual})


def test_coverage_close_to_target():
    fc = add_intervals(fake_forecasts(), alpha=0.1, window=200)
    d = fc.iloc[250:]  # skip warm-up
    covered = ((d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"])).mean()
    assert 0.86 <= covered <= 0.94


def test_no_peeking_at_unresolved_errors():
    """The first forecast has no known past errors, so its interval must be undefined."""
    fc = add_intervals(fake_forecasts(50), alpha=0.1)
    assert np.isnan(fc["upper"].iloc[0])

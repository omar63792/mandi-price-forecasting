"""mandi: forecast daily mandi (wholesale market) prices and raise price-spike alerts.

Module map, in the order the data flows through them:

    ingest.py     read price files you downloaded (or the daily API) -> DataFrame
    clean.py      fix column names, dates, units; flag bad rows
    db.py         store rows in DuckDB and build the SQL views in sql/
    series.py     turn the SQL view into one tidy daily time series per commodity/state
    models.py     forecasting models: naive, seasonal naive, LightGBM, Chronos
    backtest.py   honest "rolling origin" evaluation of those models
    conformal.py  turn point forecasts into prediction intervals
    spikes.py     define spike events and score the alerts (precision / recall)
    cli.py        the `mandi` command that runs everything
"""

__version__ = "0.1.0"

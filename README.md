# Mandi price forecasting and spike alerts: tomato, onion, potato

**The business question:** can we warn traders, procurement teams or a state
agriculture department *before or as* wholesale tomato, onion and potato
prices spike, without flooding them with false alarms?

**Headline result:** _to be filled in once the project runs on real Agmarknet
data._ Write it as one number, for example "caught X of Y spikes at Z false
alarms per market per year".

> ⚠️ Until real data is loaded, every number this repo prints comes from
> **synthetic practice data** (`mandi sample`). Never quote those numbers as results.

---

## What it does

```
 Agmarknet / data.gov.in files
            │  ingest.py, clean.py      fix columns, dates, units; flag bad rows
            ▼
     DuckDB  (sql/*.sql)                one table + SQL views → daily median price per state
            │  series.py                gap-free daily series, imputed days marked
            ▼
  Forecast 1–7 days ahead               naive · seasonal naive · LightGBM · Chronos (zero-shot)
            │  backtest.py              rolling-origin evaluation, bootstrap intervals
            ▼
  Conformal prediction intervals        conformal.py: split conformal + adaptive (ACI)
            │
            ▼
  Spike alerts, scored                  spikes.py: precision, recall, false alarms per year
            │
            ▼
  Streamlit dashboard                   app/dashboard.py
```

## Quick start (about 5 minutes, on fake data)

You need Python 3.10 or newer.

```bash
git clone https://github.com/omar63792/mandi-price-forecasting.git
cd mandi-price-forecasting
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,dashboard]"

mandi sample                     # make fake Agmarknet-style files
mandi load --sample --fresh      # clean them and store them in DuckDB
mandi backtest                   # evaluate models, intervals and alerts (~1 minute)
streamlit run app/dashboard.py   # look at the results in your browser
pytest                           # run the tests
```

Then get real data with [docs/DATA.md](docs/DATA.md), and run
`mandi load --fresh` and `mandi backtest` again.

## How the evaluation stays honest

| Risk | What this repo does |
|---|---|
| Testing on the past the model trained on | **Rolling-origin backtest**: each forecast uses only data up to its own "today" |
| A fancy model that doesn't beat a trivial one | Every error is reported **relative to the naive forecast** ("tomorrow = today") |
| Lucky noise passing as improvement | **Block-bootstrap 90% intervals** on the relative error. If the interval contains 1.0, the model is no better than naive |
| Scoring against invented values | Forward-filled (imputed) days are **never** used as ground truth |
| Intervals that look right but aren't | **Coverage** is measured: a 90% interval should contain the truth about 90% of the time |
| Spike labels defined by the model | Spikes are a fixed, model-free rule (price > 1.5× its 90-day median for 3+ days), cross-checked against known events |
| Alerts that fire all the time | A one-line baseline (+20% week-on-week) must be beaten, and **false alarms per series per year** is reported |

## Example output (synthetic data only)

Here is the format, from the fake data. Replace this section with your real results.

| model | horizon | relative MAE vs naive (90% CI) | 90% interval coverage |
|---|---|---|---|
| LightGBM | 1 day | 0.92 (0.91–0.93) | 0.90 |
| LightGBM | 7 days | 0.94 (0.88–0.99) | 0.90 |
| seasonal naive | 1 day | 1.67 (1.51–1.83) | 0.90 |

| alert rule | recall | precision | false alarms / series / year |
|---|---|---|---|
| +20% week-on-week | 0.67 | 0.14 | 1.3 |
| conformal 99% band, LightGBM | 0.67 | 0.33 | 0.7 |

The fake test year has only 3 spike events, so these precision and recall
numbers are very noisy. Real data from more states gives more events.

## Limitations (keep this section honest)

* State-level medians hide market-level detail. A spike in one big mandi may not move the state median.
* The spike rule's thresholds (1.5×, 90 days, 3 days) are a choice. Report how results change if you vary them.
* Agmarknet coverage varies: markets drop in and out of reporting, and that changes the median.
* Arrivals data is often missing; the model works without it but may be weaker.
* Conformal intervals assume the recent past resembles the near future. ACI helps, but a sudden regime change (a new export ban, for example) can still break coverage.

## Repository layout

```
src/mandi/      the Python package (start reading at __init__.py)
sql/            DuckDB schema and SQL views
tests/          pytest tests
app/            Streamlit dashboard
docs/DATA.md    how to download the real data
data/           raw files and the database (ignored by git, except daily API pulls)
reports/        backtest output (ignored by git)
```

## Background reading

* Gibbs & Candès, *Adaptive Conformal Inference Under Distribution Shift* (NeurIPS 2021)
* *A Gentle Introduction to Conformal Time Series Forecasting* (arXiv 2511.13608)
* Ansari et al., *Chronos: Learning the Language of Time Series* (2024)
* Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free online), chapters 5 and 13

## License

MIT

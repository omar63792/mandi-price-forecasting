"""All the knobs of the project live here, so you can change them in one place."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# ROOT is the repository folder (two levels above this file: src/mandi/config.py).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # put your downloaded Agmarknet files here
SAMPLE_DIR = DATA_DIR / "sample_raw" # fake practice files from `mandi sample` (kept apart from real data)
DB_PATH = DATA_DIR / "mandi.duckdb"  # the DuckDB database file (created for you)
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"       # backtest results and charts are written here

# ---------------------------------------------------------------------------
# What we study
# ---------------------------------------------------------------------------
# "TOP" = tomato, onion, potato: the three vegetables whose price spikes are a
# recurring policy problem in India (the government even runs "Operation Greens"
# for exactly these three).
COMMODITIES = ["Tomato", "Onion", "Potato"]

# ---------------------------------------------------------------------------
# Forecasting settings
# ---------------------------------------------------------------------------
HORIZON = 7            # forecast 1..7 days ahead
MAX_FFILL_DAYS = 3     # mandis close on some days; fill gaps of up to 3 days, no more
MIN_HISTORY_DAYS = 365 # a series needs at least a year of data to be modelled

# ---------------------------------------------------------------------------
# Prediction intervals and alerts
# ---------------------------------------------------------------------------
ALPHA = 0.10           # 0.10 -> 90% prediction intervals
CONFORMAL_WINDOW = 180 # how many recent forecast errors the interval is built from
ACI_GAMMA = 0.005      # step size for Adaptive Conformal Inference (see conformal.py)

# Alerts use a WIDER band than the 90% one. Why: with a 90% band, the price lands
# above the top edge about 5% of days by design, i.e. ~18 false alarms per series
# per year even when nothing is happening. A 99% band makes that ~2 a year.
ALERT_ALPHA = 0.01

# A "spike" (the ground truth we check alerts against): the price is more than
# SPIKE_RATIO times its median over the previous SPIKE_LOOKBACK days.
SPIKE_RATIO = 1.5
SPIKE_LOOKBACK = 90
SPIKE_MIN_DAYS = 3     # ignore blips shorter than 3 days
ALERT_LEAD_DAYS = 7    # an alert up to 7 days before a spike starts counts as a hit
ALERT_LAG_DAYS = 3     # ...and so does one up to 3 days after it starts

# ---------------------------------------------------------------------------
# data.gov.in API (only serves the CURRENT day, so it's for the daily update job)
# ---------------------------------------------------------------------------
# "Current Daily Price of Various Commodities from Various Markets (Mandi)"
DATA_GOV_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
DATA_GOV_API_KEY_ENV = "DATA_GOV_IN_API_KEY"  # set this environment variable

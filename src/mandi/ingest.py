"""Ingestion: get raw price rows into Python.

Two sources:

1. FILES you downloaded from agmarknet.gov.in (history). See docs/DATA.md.
   The data.gov.in API only returns *today's* prices, so history has to come
   from downloads.
2. The data.gov.in API (today's prices). Used by the daily GitHub Action to keep
   the database growing after your one-off historical download.
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from mandi.config import COMMODITIES, DATA_GOV_API_KEY_ENV, DATA_GOV_RESOURCE_ID


def read_any_table(path: Path) -> pd.DataFrame:
    """Read a CSV / Excel file. Agmarknet's 'Excel' export is sometimes really an
    HTML table with a .xls name, so if Excel reading fails we try HTML."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str)
    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path, dtype=str)
        except Exception:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return pd.read_html(StringIO(text))[0].astype(str)
    if suffix in {".html", ".htm"}:
        return pd.read_html(path)[0].astype(str)
    raise ValueError(f"Don't know how to read {path.name}")


def read_raw_folder(folder: Path) -> pd.DataFrame:
    """Read every downloaded file in a folder (and its sub-folders) into one table.

    Some Agmarknet reports omit the state column; if a file sits in a folder
    named after a state (e.g. data/raw/Maharashtra/tomato_2023.xls) we fill it in.
    """
    frames = []
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".csv", ".xls", ".xlsx", ".html", ".htm"})
    if not files:
        raise FileNotFoundError(f"No .csv/.xls/.xlsx files found in {folder}. See docs/DATA.md.")
    for path in files:
        if path.stat().st_size < 5:  # empty placeholder from fetch_history_from_api
            continue
        df = read_any_table(path)
        if df.empty:
            continue
        df["source_file"] = str(path.relative_to(folder))
        if not any(c.lower().startswith("state") for c in df.columns) and path.parent != folder:
            df["State"] = path.parent.name
        frames.append(df)
        print(f"  read {len(df):>7,} rows from {path.relative_to(folder)}")
    return pd.concat(frames, ignore_index=True)


def fetch_today_from_api(api_key: str | None = None, page_size: int = 1000) -> pd.DataFrame:
    """Download today's prices for our commodities from data.gov.in.

    Get a free API key by signing up at https://data.gov.in (My Account -> API key),
    then:  export DATA_GOV_IN_API_KEY=your_key
    """
    api_key = api_key or os.environ.get(DATA_GOV_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set the {DATA_GOV_API_KEY_ENV} environment variable first.")
    url = f"https://api.data.gov.in/resource/{DATA_GOV_RESOURCE_ID}"
    rows = []
    for commodity in COMMODITIES:
        offset = 0
        while True:  # the API returns results in pages; keep asking until a page is short
            params = {
                "api-key": api_key, "format": "json", "limit": page_size,
                "offset": offset, "filters[commodity]": commodity,
            }
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            records = resp.json().get("records", [])
            rows.extend(records)
            if len(records) < page_size:
                break
            offset += page_size
    df = pd.DataFrame(rows)
    df["source_file"] = "api.data.gov.in"
    return df


# "Variety-wise Daily Market Prices Data of Commodity" on data.gov.in. Unlike the
# "current daily price" resource above, its API documents a filter on Arrival_Date,
# so it may be able to return past days. Test it on one date before a big run.
HISTORY_RESOURCE_ID = "35985678-0d79-46b4-9ed6-6f13308a1d24"


def fetch_history_from_api(start: str, end: str, out_dir: Path, api_key: str | None = None,
                            pause_seconds: float = 1.0) -> int:
    """Download one CSV per commodity per day into out_dir/<commodity>/<YYYY-MM-DD>.csv.

    Days already downloaded are skipped, so if the run stops halfway (network,
    rate limit) just run the same command again and it carries on.
    Returns the number of rows downloaded in this run.
    """
    import time

    api_key = api_key or os.environ.get(DATA_GOV_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set the {DATA_GOV_API_KEY_ENV} environment variable first.")
    url = f"https://api.data.gov.in/resource/{HISTORY_RESOURCE_ID}"
    total = 0
    for day in pd.date_range(start, end, freq="D"):
        for commodity in COMMODITIES:
            path = out_dir / commodity.lower() / f"{day:%Y-%m-%d}.csv"
            if path.exists():
                continue
            rows, offset = [], 0
            while True:
                params = {"api-key": api_key, "format": "json", "limit": 1000, "offset": offset,
                          "filters[Commodity]": commodity, "filters[Arrival_Date]": f"{day:%d/%m/%Y}"}
                resp = requests.get(url, params=params, timeout=60)
                if resp.status_code == 429:  # "too many requests": wait and try again
                    time.sleep(30)
                    continue
                resp.raise_for_status()
                records = resp.json().get("records", [])
                rows.extend(records)
                if len(records) < 1000:
                    break
                offset += 1000
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(path, index=False)  # an empty file still marks the day as done
            total += len(rows)
            print(f"  {day:%Y-%m-%d} {commodity:<7} {len(rows):>5} rows")
            time.sleep(pause_seconds)  # be polite to a free government server
    return total

"""Cleaning: turn messy government price tables into one consistent format.

Real Agmarknet data has many problems. This module handles the common ones:

  * column names differ between downloads ("Modal Price (Rs./Quintal)" vs
    "modal_price" vs "Modal_x0020_Price")
  * dates come as "01/07/2023", "01-Jul-2023" or "2023-07-01"
  * numbers can contain commas ("1,200")
  * commodity names vary ("Tomato", "TOMATO", "Tomato(Hybrid)")
  * some rows are entered in Rs/kg instead of Rs/quintal (100x too small)
  * the same row can appear twice

Every row we keep has these columns (the "canonical schema"):

  state, district, market, commodity, variety, grade, price_date,
  min_price, max_price, modal_price   (all prices in Rs per QUINTAL, 1 quintal = 100 kg)
  arrivals_tonnes                     (quantity that reached the market; often missing)
  quality_flag                        ("ok" or a short reason the row is suspect)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mandi.config import COMMODITIES

CANONICAL_COLUMNS = [
    "state", "district", "market", "commodity", "variety", "grade", "price_date",
    "min_price", "max_price", "modal_price", "arrivals_tonnes", "quality_flag",
]

# Every spelling we have seen for each canonical column. We compare names after
# "squashing" them (lowercase, letters/digits only), so "Min Price (Rs./Quintal)"
# becomes "minpricersquintal". Add new spellings here if a download fails.
COLUMN_ALIASES = {
    "state": ["state", "statename"],
    "district": ["district", "districtname"],
    "market": ["market", "marketname", "marketcenter", "mandi", "apmc"],
    "commodity": ["commodity", "commodityname"],
    "variety": ["variety", "varietyname"],
    "grade": ["grade"],
    "price_date": ["pricedate", "arrivaldate", "reporteddate", "date"],
    "min_price": ["minprice", "minpricersquintal", "minimumprice", "minx0020price"],
    "max_price": ["maxprice", "maxpricersquintal", "maximumprice", "maxx0020price"],
    "modal_price": ["modalprice", "modalpricersquintal", "modalx0020price"],
    "arrivals_tonnes": ["arrivals", "arrivalstonnes", "arrival", "arrivalsquantity"],
}


def _squash(name: str) -> str:
    """'Modal Price (Rs./Quintal)' -> 'modalpricersquintal'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename whatever columns the file has to our canonical names."""
    lookup = {alias: canon for canon, aliases in COLUMN_ALIASES.items() for alias in aliases}
    rename = {}
    for col in df.columns:
        canon = lookup.get(_squash(col))
        if canon and canon not in rename.values():
            rename[col] = canon
    df = df.rename(columns=rename)

    required = ["market", "commodity", "price_date", "modal_price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find columns {missing}. The file has: {list(df.columns)}. "
            "Add the new spelling to COLUMN_ALIASES in src/mandi/clean.py."
        )
    # Optional columns that are absent become empty columns.
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df


def parse_dates(s: pd.Series) -> pd.Series:
    """Parse Indian-style dates. dayfirst=True means 01/07/2023 is 1 July, not 7 Jan."""
    return pd.to_datetime(s, dayfirst=True, errors="coerce", format="mixed").dt.normalize()


def parse_numbers(s: pd.Series) -> pd.Series:
    """'1,200' -> 1200.0 ; '-', 'NR' or blank -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def normalise_commodity(s: pd.Series) -> pd.Series:
    """'TOMATO', 'Tomato(Hybrid)', ' onion ' -> 'Tomato', 'Tomato', 'Onion'.
    Anything that isn't one of our commodities is kept as-is (and later ignored)."""
    out = s.astype(str).str.strip()
    for name in COMMODITIES:
        out = out.mask(out.str.lower().str.startswith(name.lower()), name)
    return out


def flag_quality(df: pd.DataFrame) -> pd.Series:
    """Give each row a reason it looks wrong, or 'ok'.

    We FLAG instead of DELETE so you can count how much data each rule removes
    and mention it in your README. Interviewers like that.
    """
    flag = pd.Series("ok", index=df.index)
    flag[df["modal_price"].isna() | (df["modal_price"] <= 0)] = "missing_or_zero_price"
    # Tomato/onion/potato have essentially never sold below Rs 1/kg (= Rs 100/quintal)
    # in recent years. A value like 25 is almost certainly Rs 25/kg typed into a
    # Rs/quintal field.
    flag[(flag == "ok") & (df["modal_price"] < 100)] = "suspect_unit_rs_per_kg"
    # Above Rs 300/kg is not plausible for wholesale tomato/onion/potato.
    flag[(flag == "ok") & (df["modal_price"] > 30_000)] = "implausibly_high"
    # Modal (most common) price should sit between min and max. Allow 5% slack.
    bad_order = (df["min_price"].notna() & df["max_price"].notna() & (
        (df["modal_price"] < 0.95 * df["min_price"]) | (df["modal_price"] > 1.05 * df["max_price"])
    ))
    flag[(flag == "ok") & bad_order] = "modal_outside_min_max"
    flag[df["price_date"].isna()] = "bad_date"
    return flag


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: columns -> types -> names -> flags -> de-duplicate."""
    df = standardise_columns(df.copy())
    df["price_date"] = parse_dates(df["price_date"])
    for col in ["min_price", "max_price", "modal_price", "arrivals_tonnes"]:
        df[col] = parse_numbers(df[col])
    for col in ["state", "district", "market", "variety", "grade"]:
        df[col] = df[col].astype("string").str.strip().str.title()
    df["commodity"] = normalise_commodity(df["commodity"])

    df = df[df["commodity"].isin(COMMODITIES)]
    df["quality_flag"] = flag_quality(df)

    # Same market + commodity + variety + grade + date reported twice -> keep the last.
    key = ["market", "commodity", "variety", "grade", "price_date"]
    df = df.drop_duplicates(subset=key, keep="last")
    # keep the name of the file each row came from, if ingest.py recorded it
    extra = ["source_file"] if "source_file" in df.columns else []
    return df[CANONICAL_COLUMNS + extra].reset_index(drop=True)

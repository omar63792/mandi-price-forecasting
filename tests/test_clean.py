"""Tests for clean.py. Run all tests with:  pytest"""

import pandas as pd

from mandi.clean import clean


def raw_rows():
    return pd.DataFrame({
        "State": ["Maharashtra"] * 4,
        "District Name": ["Pune"] * 4,
        "Market Name": ["Pune"] * 4,
        "Commodity": ["TOMATO", "Tomato", "Tomato(Hybrid)", "Brinjal"],
        "Variety": ["Hybrid", "Hybrid", "Local", "Other"],
        "Grade": ["FAQ"] * 4,
        "Min Price (Rs./Quintal)": ["1,000", "1,000", "10", "500"],
        "Max Price (Rs./Quintal)": ["1,400", "1,400", "14", "700"],
        "Modal Price (Rs./Quintal)": ["1,200", "1,200", "12", "600"],
        "Price Date": ["01/07/2023", "01/07/2023", "02-Jul-2023", "01/07/2023"],
    })


def test_columns_dates_numbers():
    df = clean(raw_rows())
    assert {"market", "modal_price", "price_date"} <= set(df.columns)
    assert df["price_date"].min() == pd.Timestamp("2023-07-01")  # day-first, not 7 January
    assert df["modal_price"].max() == 1200.0                      # "1,200" parsed


def test_commodity_filter_and_dedupe():
    df = clean(raw_rows())
    assert set(df["commodity"]) == {"Tomato"}   # brinjal dropped, spellings unified
    assert len(df) == 2                         # the duplicated row appears once


def test_unit_error_is_flagged_not_deleted():
    df = clean(raw_rows())
    flags = dict(zip(df["variety"], df["quality_flag"]))
    assert flags["Local"] == "suspect_unit_rs_per_kg"
    assert flags["Hybrid"] == "ok"

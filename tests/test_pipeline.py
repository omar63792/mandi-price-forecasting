"""End-to-end smoke test on a small slice of the fake data."""

from mandi.clean import clean
from mandi.db import connect, load_daily, upsert_prices
from mandi.ingest import read_raw_folder
from mandi.sample import make_sample
from mandi.series import make_panel


def test_files_to_panel(tmp_path):
    make_sample(tmp_path / "raw", start="2022-01-01", end="2023-06-30")
    con = connect(tmp_path / "test.duckdb")
    upsert_prices(con, clean(read_raw_folder(tmp_path / "raw")))
    upsert_prices(con, clean(read_raw_folder(tmp_path / "raw")))  # loading twice must not duplicate
    n = con.execute("SELECT count(*) FROM raw_prices").fetchone()[0]
    panel = make_panel(load_daily(con, "state"))
    assert panel["series_id"].nunique() == 9
    assert n > 0 and panel["price"].notna().mean() > 0.95

"""The `mandi` command. Run `mandi --help` to see the steps.

Typical first run (on fake data):
    mandi sample              # write fake Agmarknet files to data/sample_raw/
    mandi load --sample       # clean them and store in data/mandi.duckdb
    mandi backtest    # evaluate models, write reports/
Then download real data (docs/DATA.md) and run:  mandi load --fresh  and  mandi backtest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mandi import config
from mandi.clean import clean
from mandi.db import connect, load_daily, upsert_prices


def cmd_sample(args):
    from mandi.sample import make_sample

    paths = make_sample(config.SAMPLE_DIR)
    print(f"Wrote {len(paths)} fake files to {config.SAMPLE_DIR}. Next: mandi load --sample")


def cmd_load(args):
    from mandi.ingest import read_raw_folder

    folder = config.SAMPLE_DIR if args.sample else Path(args.folder)
    if args.fresh and config.DB_PATH.exists():
        config.DB_PATH.unlink()  # start from an empty database
        print(f"Deleted old {config.DB_PATH.name}")
    print(f"Reading files from {folder} ...")
    raw = read_raw_folder(folder)
    cleaned = clean(raw)
    con = connect()
    n = upsert_prices(con, cleaned)
    print(f"Stored {n:,} cleaned rows (from {len(raw):,} raw rows).")
    print("\nData-quality summary (rows flagged by clean.py):")
    print(con.execute("SELECT * FROM quality_report").df().to_string(index=False))


def cmd_fetch_today(args):
    from mandi.ingest import fetch_today_from_api

    raw = fetch_today_from_api()
    if raw.empty:
        print("API returned no rows today.")
        return
    out = config.RAW_DIR / "daily" / f"{pd.Timestamp.today():%Y-%m-%d}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out, index=False)
    con = connect()
    n = upsert_prices(con, clean(raw))
    print(f"Saved {out} and stored {n:,} rows.")


def cmd_fetch_history(args):
    from mandi.ingest import fetch_history_from_api

    out = config.RAW_DIR / "api_history"
    n = fetch_history_from_api(args.start, args.end, out)
    print(f"Downloaded {n:,} rows into {out}. Next: mandi load")


def cmd_backtest(args):
    from mandi.backtest import accuracy_table, run_backtest
    from mandi.conformal import add_intervals, coverage_report
    from mandi.series import make_panel
    from mandi.spikes import baseline_alerts, conformal_alerts, label_spike_events, score_alerts

    con = connect()
    panel = make_panel(load_daily(con, args.level))
    print(f"{panel['series_id'].nunique()} series, {panel['date'].min():%Y-%m-%d} to {panel['date'].max():%Y-%m-%d}")

    fc = run_backtest(panel, args.models, test_days=args.test_days)
    fc = pd.concat([add_intervals(g).assign(model=m) for m, g in fc.groupby("model")], ignore_index=True)

    out = config.REPORTS_DIR
    out.mkdir(exist_ok=True)
    fc.to_parquet(out / "forecasts.parquet")

    acc = accuracy_table(fc)
    cov = coverage_report(fc[fc["is_test"]])

    test = fc[fc["is_test"]]
    start, end = test["origin"].min(), test["target_date"].max()
    events = label_spike_events(panel)
    n_series = panel["series_id"].nunique()
    alert_rows = [{"alert_rule": "baseline: +20% week-on-week",
                   **score_alerts(baseline_alerts(panel), events, start, end, n_series)}]
    # alerts use a separate, wider (99%) band built from the 1-day-ahead forecasts
    h1 = fc[fc["horizon"] == 1].drop(columns=["lower", "upper", "alpha_used", "log_err"])
    alert_fc = pd.concat([add_intervals(g, alpha=config.ALERT_ALPHA).assign(model=m)
                          for m, g in h1.groupby("model")], ignore_index=True)
    band = round(100 * (1 - config.ALERT_ALPHA))
    for m in args.models:
        alert_rows.append({"alert_rule": f"conformal {band}% band, {m}",
                           **score_alerts(conformal_alerts(alert_fc, m), events, start, end, n_series)})
    alerts = pd.DataFrame(alert_rows)

    acc.to_csv(out / "accuracy.csv", index=False)
    cov.to_csv(out / "coverage.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)
    events.to_csv(out / "spike_events.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\nACCURACY (test period; rel_mae < 1 means better than naive)")
    print(acc[acc["horizon"].isin([1, 7])].to_string(index=False))
    print("\nINTERVAL COVERAGE (target 0.90)")
    print(cov[cov["horizon"].isin([1, 7])].to_string(index=False))
    print("\nSPIKE ALERTS")
    print(alerts.to_string(index=False))
    print(f"\nAll tables written to {out}/")


def main(argv=None):
    p = argparse.ArgumentParser(prog="mandi", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("sample", help="write fake Agmarknet-style files (for practice and tests)").set_defaults(func=cmd_sample)

    s = sub.add_parser("load", help="clean downloaded files and store them in DuckDB")
    s.add_argument("folder", nargs="?", default=str(config.RAW_DIR), help="folder of downloaded files")
    s.add_argument("--sample", action="store_true", help="load the fake practice files instead")
    s.add_argument("--fresh", action="store_true",
                   help="delete the database first (do this when switching from fake to real data)")
    s.set_defaults(func=cmd_load)

    sub.add_parser("fetch-today", help="pull today's prices from the data.gov.in API").set_defaults(func=cmd_fetch_today)

    s = sub.add_parser("fetch-history", help="try downloading past days from the data.gov.in API")
    s.add_argument("--start", required=True, help="first day, e.g. 2023-01-01")
    s.add_argument("--end", required=True, help="last day, e.g. 2023-12-31")
    s.set_defaults(func=cmd_fetch_history)

    s = sub.add_parser("backtest", help="evaluate models, intervals and spike alerts")
    s.add_argument("--models", nargs="+", default=["naive", "seasonal_naive_7", "lightgbm"],
                   help="any of: naive seasonal_naive_7 lightgbm chronos_bolt_small")
    s.add_argument("--level", choices=["state", "national"], default="state")
    s.add_argument("--test-days", type=int, default=365)
    s.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

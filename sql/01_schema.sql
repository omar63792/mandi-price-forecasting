-- 01_schema.sql : the one table that stores every price row we have ever loaded.
-- All prices are Rs per QUINTAL (100 kg), exactly as Agmarknet reports them.

CREATE TABLE IF NOT EXISTS raw_prices (
    state            VARCHAR,
    district         VARCHAR,
    market           VARCHAR NOT NULL,
    commodity        VARCHAR NOT NULL,
    variety          VARCHAR,
    grade            VARCHAR,
    price_date       DATE    NOT NULL,
    min_price        DOUBLE,
    max_price        DOUBLE,
    modal_price      DOUBLE,
    arrivals_tonnes  DOUBLE,
    quality_flag     VARCHAR,          -- 'ok' or why the row is suspect (see clean.py)
    source_file      VARCHAR,          -- which download the row came from
    loaded_at        TIMESTAMP DEFAULT current_timestamp,
    -- One row per market / commodity / variety / grade / day. Re-loading the same
    -- file therefore replaces rows instead of duplicating them.
    PRIMARY KEY (market, commodity, variety, grade, price_date)
);

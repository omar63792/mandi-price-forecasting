-- 02_views.sql : SQL that turns raw rows into model-ready daily series.
-- Good interview material: window functions, robust aggregation, data-quality filters.

-- Step 1. Keep good rows, convert to Rs/kg, and drop prices that are wildly off
-- that market's own recent level (typos such as an extra zero).
CREATE OR REPLACE VIEW clean_prices AS
WITH ok_rows AS (
    SELECT
        state, district, market, commodity, variety, price_date,
        modal_price / 100.0 AS price_rs_kg,   -- Rs/quintal -> Rs/kg
        arrivals_tonnes
    FROM raw_prices
    WHERE quality_flag = 'ok'
),
-- A market can report several varieties/grades on one day; take the median.
market_day AS (
    SELECT state, district, market, commodity, price_date,
           median(price_rs_kg)    AS price_rs_kg,
           sum(arrivals_tonnes)   AS arrivals_tonnes
    FROM ok_rows
    GROUP BY state, district, market, commodity, price_date
),
with_ref AS (
    SELECT *,
        -- median of this market's previous 30 rows (window function)
        median(price_rs_kg) OVER (
            PARTITION BY market, commodity ORDER BY price_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS ref_price
    FROM market_day
)
SELECT state, district, market, commodity, price_date, price_rs_kg, arrivals_tonnes
FROM with_ref
-- keep a row unless it is more than 5x above or below the market's own recent median
WHERE ref_price IS NULL OR price_rs_kg BETWEEN ref_price / 5 AND ref_price * 5;

-- Step 2. One number per state per commodity per day: the median across markets.
-- Median, not mean, so one strange mandi cannot move the whole state.
CREATE OR REPLACE VIEW state_daily AS
SELECT state, commodity, price_date,
       median(price_rs_kg)       AS price_rs_kg,
       count(DISTINCT market)    AS n_markets,
       sum(arrivals_tonnes)      AS arrivals_tonnes
FROM clean_prices
GROUP BY state, commodity, price_date;

-- Step 3. The national series: same idea across all markets in India.
CREATE OR REPLACE VIEW national_daily AS
SELECT 'All India' AS state, commodity, price_date,
       median(price_rs_kg)       AS price_rs_kg,
       count(DISTINCT market)    AS n_markets,
       sum(arrivals_tonnes)      AS arrivals_tonnes
FROM clean_prices
GROUP BY commodity, price_date;

-- Step 4. A data-quality report you can paste into the README.
CREATE OR REPLACE VIEW quality_report AS
SELECT commodity, quality_flag, count(*) AS n_rows,
       round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY commodity), 2) AS pct
FROM raw_prices
GROUP BY commodity, quality_flag
ORDER BY commodity, n_rows DESC;

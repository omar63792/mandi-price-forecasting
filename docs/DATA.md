# Getting the real price data

The code can't download history for you automatically from Agmarknet, because
its reports sit behind a captcha. There are two routes. **Try Route A first**:
if it works, it is far less clicking.

All prices on these sites are in **Rs per quintal** (1 quintal = 100 kg). The
code converts them to Rs/kg for you.

---

## Route A: the data.gov.in API (try this first)

data.gov.in publishes a dataset called **"Variety-wise Daily Market Prices Data
of Commodity"**. Its API lets you filter by `Arrival_Date`, which suggests it can
return past days. I couldn't confirm how far back it goes, so test it with one
day before starting a big download.

1. **Get a free API key.** Go to <https://data.gov.in>, click **Login | Register**,
   create an account, then open **My Account** and copy your API key.
   (Keep the key private. Don't paste it into code you push to GitHub.)
2. **Put the key in an environment variable** in the terminal you will use:
   * Mac/Linux: `export DATA_GOV_IN_API_KEY=paste_your_key_here`
   * Windows PowerShell: `$env:DATA_GOV_IN_API_KEY="paste_your_key_here"`
3. **Test one day:**
   ```bash
   mandi fetch-history --start 2023-07-15 --end 2023-07-15
   ```
   You should see lines like `2023-07-15 Tomato   850 rows`. If you see `0 rows`
   for every commodity, the API doesn't serve that date. Use Route B instead.
4. **Download the range you want** (it skips days it already has, so you can
   stop and re-run any time):
   ```bash
   mandi fetch-history --start 2021-01-01 --end 2025-12-31
   ```
   Three commodities over five years is about 5,500 requests. Expect a few hours.
   If the API says "too many requests", the code waits 30 seconds and continues.
5. **Load it:** `mandi load --fresh`

---

## Route B: manual downloads from Agmarknet 2.0

1. Open <https://agmarknet.gov.in>, then go to **Reports → Price & Arrival Reports**.
2. Choose **"Date Wise Prices For Specified Commodity"**.
3. Pick a **Year**, a **Month**, a **State/UT** and a **Commodity** (Tomato,
   Onion or Potato), type the captcha, and fetch the report.
4. Use the **Download** button to save the file.
5. Save it under `data/raw/<State name>/`, for example
   `data/raw/Maharashtra/tomato_2023_07.xls`. The folder name is used as the
   state if the file itself has no State column.
6. Repeat for each month, state and commodity you want, then run `mandi load --fresh`.

**Keep the scope small, so it stays doable.** That's one download per month per
state per commodity. A sensible first target:

| Choice | Suggestion | Why |
|---|---|---|
| Commodities | Tomato, Onion, Potato | the three with recurring spikes |
| States | Maharashtra, Karnataka, Uttar Pradesh | big producers in different regions |
| Years | 2022 to 2025 | covers the July 2023 tomato and late-2023 onion spikes |

That is 3 × 3 × 48 = 432 files. Start with **one commodity and one state**
(48 files), get the whole pipeline working, and add more later.

The site layout can change. If the menu names differ, look for any report that
gives daily prices by market for one commodity and a date range.

### If `mandi load` says it can't find columns

The error message lists the columns your file actually has. Open
`src/mandi/clean.py`, find `COLUMN_ALIASES`, and add the new spelling to the
right list. Use lowercase letters and digits only, for example
`"modalpricersqtl"`. That is a normal part of working with government data.

---

## Keeping it up to date (optional)

`.github/workflows/daily_ingest.yml` runs every evening on GitHub, downloads
that day's prices with the API and commits them to `data/raw/daily/`. To switch
it on, open your repo on GitHub and go to **Settings → Secrets and variables →
Actions → New repository secret**. Name it `DATA_GOV_IN_API_KEY` and paste your key.

---

## Check that the dates are real

`data/known_events.csv` lists three well-known spikes: onion in late 2019,
tomato in July 2023, and onion in late 2023. After loading real data, check
that `reports/spike_events.csv` finds spikes near those dates. If it doesn't,
something in the data or the spike rule needs a look. The dates in that file
are approximate, so confirm each one against a news or PIB source before you
quote it.

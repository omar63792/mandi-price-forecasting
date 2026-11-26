"""Make FAKE Agmarknet-style files so you can run the whole pipeline before
downloading real data, and so the tests have something to chew on.

The fake data is deliberately messy in the same ways real data is (missing
Sundays, duplicates, Rs/kg typos, varieties, different date formats), and it
contains injected price spikes, so the spike detector has something to find.

NEVER report results computed on this data as real results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STATES = {"Maharashtra": ["Pune", "Nashik", "Lasalgaon"],
          "Karnataka": ["Bangalore", "Kolar", "Mysore"],
          "Uttar Pradesh": ["Agra", "Lucknow", "Kanpur"]}
BASE_PRICE = {"Tomato": 1500, "Onion": 1800, "Potato": 1200}  # Rs/quintal
SEASON_AMP = {"Tomato": 0.35, "Onion": 0.25, "Potato": 0.15}


def make_sample(out_dir: Path, start: str = "2021-01-01", end: str = "2024-12-31", seed: int = 42) -> list[Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    written = []
    for commodity, base in BASE_PRICE.items():
        # one shared national shock path per commodity (weather, festivals, policy)
        shock = np.zeros(n)
        for _ in range(int(len(dates) / 365 * 1.5)):  # ~1.5 spikes a year
            s = rng.integers(120, n - 60)
            length = rng.integers(20, 45)
            height = rng.uniform(0.5, 1.0)  # +65% .. +170% at peak
            ramp = np.sin(np.linspace(0, np.pi, length)) * height
            shock[s:s + length] += ramp[: max(0, min(length, n - s))]
        season = SEASON_AMP[commodity] * np.sin(2 * np.pi * (dates.dayofyear.values - 200) / 365.25)
        for state, markets in STATES.items():
            # slow random wander per state (AR(1) on log scale)
            ar = np.zeros(n)
            for t in range(1, n):
                ar[t] = 0.98 * ar[t - 1] + rng.normal(0, 0.02)
            state_level = rng.uniform(0.85, 1.15)
            rows = []
            for market in markets:
                mkt_level = rng.uniform(0.9, 1.1)
                lp = np.log(base * state_level * mkt_level) + season + ar + shock + rng.normal(0, 0.04, n)
                modal = np.exp(lp)
                for i, d in enumerate(dates):
                    if d.dayofweek == 6 or rng.random() < 0.08:   # closed Sundays + random gaps
                        continue
                    m = round(modal[i], -1)
                    mn, mx = round(m * rng.uniform(0.75, 0.95), -1), round(m * rng.uniform(1.05, 1.25), -1)
                    if rng.random() < 0.002:   # Rs/kg typo
                        m = round(m / 100)
                    variety = "Hybrid" if commodity == "Tomato" and rng.random() < 0.5 else "Other"
                    date_str = d.strftime("%d/%m/%Y") if rng.random() < 0.9 else d.strftime("%d-%b-%Y")
                    rows.append({
                        "State": state, "District Name": market, "Market Name": market,
                        "Commodity": commodity.upper() if rng.random() < 0.1 else commodity,
                        "Variety": variety, "Grade": "FAQ",
                        "Min Price (Rs./Quintal)": f"{mn:,.0f}", "Max Price (Rs./Quintal)": f"{mx:,.0f}",
                        "Modal Price (Rs./Quintal)": f"{m:,.0f}",
                        "Arrivals (Tonnes)": round(rng.gamma(4, 25) * np.exp(-shock[i]), 1),
                        "Price Date": date_str,
                    })
            df = pd.DataFrame(rows)
            df = pd.concat([df, df.sample(frac=0.01, random_state=1)])  # duplicate rows
            path = out_dir / state / f"{commodity.lower()}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            written.append(path)
    return written

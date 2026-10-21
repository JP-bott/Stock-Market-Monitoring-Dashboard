"""Generate the SYNTHETIC demonstration dataset used by the dashboard.

Run from the project root:

    python scripts/generate_sample_data.py

The output (data/sample_stock_prices.csv) is fully reproducible: a fixed
random seed is used, so every run writes the same file.

IMPORTANT: all companies, tickers and prices below are fictional. The numbers
come from a random-walk model and must never be read as real market quotes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

SEED = 42
START_DATE = "2025-01-02"
END_DATE = "2026-06-30"  # about 18 months of trading days
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_stock_prices.csv"

# Fictional companies. mu = expected annual drift (first / second half of the
# sample), vol = annual volatility, rho = correlation with the "market"
# factor, beta = sensitivity to market regimes, div = annual dividend yield.
COMPANIES = [
    dict(ticker="NVLX", name="Novalux Semiconductors", start=118.0, mu=(0.55, 0.35), vol=0.46, rho=0.65, beta=1.4, base_vol=21_000_000, div=0.0),
    dict(ticker="ARCD", name="Arcadia Software", start=212.0, mu=(0.30, 0.22), vol=0.30, rho=0.60, beta=1.1, base_vol=9_000_000, div=0.0),
    dict(ticker="BLWK", name="Bluewick Retail Group", start=64.0, mu=(0.02, 0.08), vol=0.24, rho=0.45, beta=0.8, base_vol=6_500_000, div=0.012),
    dict(ticker="CRST", name="Crestline Energy", start=88.0, mu=(-0.25, 0.30), vol=0.34, rho=0.35, beta=0.9, base_vol=7_800_000, div=0.03),
    dict(ticker="DLTN", name="Deltaon Biotech", start=27.0, mu=(0.10, -0.05), vol=0.68, rho=0.30, beta=1.2, base_vol=14_000_000, div=0.0),
    dict(ticker="FRGE", name="Forgeline Industrials", start=142.0, mu=(0.10, 0.12), vol=0.19, rho=0.55, beta=0.9, base_vol=2_400_000, div=0.018),
    dict(ticker="GLDW", name="Goldwater Utilities", start=71.0, mu=(0.06, 0.05), vol=0.13, rho=0.25, beta=0.4, base_vol=3_100_000, div=0.038),
    dict(ticker="HBRN", name="Harborne Financial", start=96.0, mu=(0.14, 0.10), vol=0.25, rho=0.70, beta=1.0, base_vol=11_500_000, div=0.022),
    dict(ticker="KMTO", name="Kometo Telecom", start=39.0, mu=(-0.08, -0.12), vol=0.21, rho=0.40, beta=0.7, base_vol=12_800_000, div=0.045),
    dict(ticker="LYRA", name="Lyra Media Networks", start=53.0, mu=(0.12, 0.00), vol=0.52, rho=0.50, beta=1.3, base_vol=8_200_000, div=0.0),
]


def trading_days() -> pd.DatetimeIndex:
    """Weekdays excluding US federal holidays (exchange-specific closures are not modeled)."""
    cbd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    return pd.date_range(START_DATE, END_DATE, freq=cbd)


def market_regime(n: int) -> np.ndarray:
    """Daily drift shared by all stocks: calm growth, a sell-off, then a recovery."""
    regime = np.full(n, 0.0003)
    regime[200:245] = -0.0030  # synthetic sell-off
    regime[245:310] = 0.0016  # synthetic rebound
    return regime


def simulate_company(cfg: dict, dates: pd.DatetimeIndex, market_z: np.ndarray,
                     regime: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(dates)
    sigma = cfg["vol"] / np.sqrt(252)
    half = n // 2
    mu_daily = np.where(np.arange(n) < half, cfg["mu"][0], cfg["mu"][1]) / 252

    eps = rng.standard_normal(n)
    z = cfg["rho"] * market_z + np.sqrt(1 - cfg["rho"] ** 2) * eps
    returns = mu_daily - 0.5 * sigma**2 + cfg["beta"] * regime + sigma * z

    close = cfg["start"] * np.exp(np.cumsum(returns))
    prev_close = np.concatenate([[cfg["start"]], close[:-1]])

    open_ = prev_close * np.exp(rng.normal(0, 0.25 * sigma, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.45 * sigma, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.45 * sigma, n)))

    # Volume rises on big moves (|z| large) and has random day-to-day noise
    shock = np.abs(returns - mu_daily) / sigma
    volume = cfg["base_vol"] * rng.lognormal(0, 0.25, n) * (1 + 0.5 * shock)

    # Adjusted close: assume a quarterly dividend (~every 63 trading days)
    adj_factor = np.ones(n)
    if cfg["div"] > 0:
        ex_days = np.arange(20, n, 63)
        per_payment = 1 - cfg["div"] / 4
        for i in range(n):
            future = int((ex_days > i).sum())
            adj_factor[i] = per_payment**future

    df = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": cfg["ticker"],
            "Company": cfg["name"],
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Adjusted_Close": np.round(close * adj_factor, 2),
            "Volume": np.round(volume).astype("int64"),
        }
    )
    # Rounding can break OHLC rules by a cent; repair so High/Low always bracket the day
    df["High"] = df[["Open", "High", "Low", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "High", "Low", "Close"]].min(axis=1)
    return df


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = trading_days()
    market_z = rng.standard_normal(len(dates))
    regime = market_regime(len(dates))
    frames = [simulate_company(c, dates, market_z, regime, rng) for c in COMPANIES]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def main() -> None:
    df = generate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(df):,} rows for {df['Ticker'].nunique()} synthetic tickers to {OUTPUT_PATH}")
    print(f"Date range: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")
    print("NOTE: this is synthetic demonstration data, not real market prices.")


if __name__ == "__main__":
    main()

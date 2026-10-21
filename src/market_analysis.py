"""Reusable market-analysis helpers for the Stock Market Monitoring Dashboard.

Everything here uses only pandas and NumPy (plus an optional, lazily imported
yfinance call), so the same functions can be used by the Streamlit app, the
Jupyter notebook and the unit tests.

All indicator functions work on plain pandas Series or DataFrames and return
NaN (not made-up numbers) when there is not enough history.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "Date",
    "Ticker",
    "Company",
    "Open",
    "High",
    "Low",
    "Close",
    "Adjusted_Close",
    "Volume",
]
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adjusted_Close"]
TRADING_DAYS_PER_YEAR = 252

# Label shown in the app -> period string understood by yfinance
PERIOD_OPTIONS = {
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "2 years": "2y",
    "5 years": "5y",
}

# 1-10 characters: letters, digits, dot or dash (e.g. AAPL, BRK-B)
_TICKER_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


@dataclass
class DataResult:
    """Outcome of loading price data for one ticker (sample or live)."""

    data: Optional[pd.DataFrame]
    error: Optional[str] = None
    company: str = ""
    fetched_at: Optional[str] = None  # UTC timestamp for live data, else None


# --------------------------------------------------------------------------
# Loading, validating and cleaning
# --------------------------------------------------------------------------


def normalize_ticker(text: object) -> Optional[str]:
    """Return an upper-case ticker, or None if the text is not a valid symbol."""
    if text is None:
        return None
    symbol = str(text).strip().upper()
    return symbol if _TICKER_PATTERN.match(symbol) else None


def load_sample_data(path: str | Path) -> pd.DataFrame:
    """Load the bundled synthetic CSV, parse dates and sort by ticker/date."""
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sample data is missing columns: {missing}")
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def validate_price_data(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable data-quality problems (empty = clean)."""
    issues: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"Missing columns: {missing}"]

    na_counts = df[REQUIRED_COLUMNS].isna().sum()
    for col, n in na_counts.items():
        if n:
            issues.append(f"{n} missing values in '{col}'")

    for col in PRICE_COLUMNS:
        bad = int((df[col] <= 0).sum())
        if bad:
            issues.append(f"{bad} non-positive values in '{col}'")
    if (df["Volume"] < 0).any():
        issues.append(f"{int((df['Volume'] < 0).sum())} negative Volume values")

    high_bad = df["High"] < df[["Open", "Close", "Low"]].max(axis=1)
    low_bad = df["Low"] > df[["Open", "Close", "High"]].min(axis=1)
    if high_bad.any():
        issues.append(f"{int(high_bad.sum())} rows where High is below Open/Close/Low")
    if low_bad.any():
        issues.append(f"{int(low_bad.sum())} rows where Low is above Open/Close/High")

    dup = int(df.duplicated(["Ticker", "Date"]).sum())
    if dup:
        issues.append(f"{dup} duplicate Ticker/Date rows")
    unsorted = [t for t, s in df.groupby("Ticker")["Date"] if not s.is_monotonic_increasing]
    if unsorted:
        issues.append(f"Dates not sorted for tickers: {unsorted}")
    return issues


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and unusable rows, fill a missing Adjusted_Close, sort."""
    out = df.copy()
    out = out.drop_duplicates(["Ticker", "Date"], keep="last")
    out = out.dropna(subset=["Date", "Close"])
    out = out[out["Close"] > 0]
    for col in ["Open", "High", "Low"]:
        out[col] = out[col].fillna(out["Close"])
    out["Adjusted_Close"] = out["Adjusted_Close"].fillna(out["Close"])
    out["Volume"] = out["Volume"].fillna(0).clip(lower=0)
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def filter_dates(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Keep rows whose Date is between start and end (inclusive)."""
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] <= pd.Timestamp(end))
    return df.loc[mask]


# --------------------------------------------------------------------------
# Live data (optional, via yfinance)
# --------------------------------------------------------------------------


def _yfinance_history(symbol: str, period: str):
    """Download daily history from Yahoo Finance via yfinance.

    yfinance is imported here (not at module level) so that sample mode keeps
    working even if the package is missing or the network is down.
    """
    import yfinance as yf  # deliberate lazy import

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, auto_adjust=False, actions=False)
    name = None
    try:  # company name is a nice-to-have; never fail because of it
        info = ticker.info
        name = info.get("shortName") or info.get("longName")
    except Exception:  # noqa: BLE001
        name = None
    return hist, name


def _standardize_live(hist: pd.DataFrame, symbol: str, company: str) -> pd.DataFrame:
    """Convert a yfinance history frame to the project's column layout."""
    hist = hist.copy()
    if isinstance(hist.columns, pd.MultiIndex):  # newer yfinance versions
        hist.columns = hist.columns.get_level_values(0)

    dates = pd.to_datetime(hist.index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize()

    def col(name: str, fallback: Optional[str] = None) -> pd.Series:
        if name in hist.columns:
            source = hist[name]
        elif fallback is not None and fallback in hist.columns:
            source = hist[fallback]
        else:
            source = pd.Series(np.nan, index=hist.index)
        return pd.to_numeric(source, errors="coerce").to_numpy()

    out = pd.DataFrame(
        {
            "Date": dates,
            "Ticker": symbol,
            "Company": company,
            "Open": col("Open"),
            "High": col("High"),
            "Low": col("Low"),
            "Close": col("Close"),
            "Adjusted_Close": col("Adj Close", fallback="Close"),
            "Volume": col("Volume"),
        }
    )
    return clean_price_data(out)


def fetch_live_data(
    ticker: str,
    period: str,
    fetcher: Optional[Callable] = None,
) -> DataResult:
    """Fetch live historical data and never raise: errors come back as text.

    ``fetcher`` can be replaced in tests so no network is needed.
    """
    symbol = normalize_ticker(ticker)
    if symbol is None:
        return DataResult(None, f"'{ticker}' is not a valid ticker symbol format.")
    if period not in PERIOD_OPTIONS.values():
        return DataResult(None, f"Unsupported period '{period}'.")

    fetcher = fetcher or _yfinance_history
    try:
        hist, name = fetcher(symbol, period)
    except ImportError:
        return DataResult(
            None, "The 'yfinance' package is not installed (pip install yfinance)."
        )
    except Exception as exc:  # noqa: BLE001 - network errors, rate limits, etc.
        return DataResult(
            None,
            f"Could not retrieve live data for {symbol} ({type(exc).__name__}). "
            "This can be a network problem or a rate limit; try again later.",
        )

    if hist is None or len(hist) == 0:
        return DataResult(
            None,
            f"No data was returned for '{symbol}'. Check the symbol or try again later.",
        )

    try:
        df = _standardize_live(hist, symbol, name or symbol)
    except Exception as exc:  # noqa: BLE001
        return DataResult(
            None, f"Live data for {symbol} could not be read ({type(exc).__name__})."
        )
    if df.empty:
        return DataResult(None, f"'{symbol}' returned no usable price rows.")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return DataResult(df, None, name or symbol, stamp)


# --------------------------------------------------------------------------
# Returns, moving averages and indicators
# --------------------------------------------------------------------------


def daily_returns(close: pd.Series) -> pd.Series:
    """Simple daily return: Close_t / Close_(t-1) - 1 (NaN for the first row)."""
    close = pd.Series(close, dtype="float64")
    prev = close.shift(1).replace(0, np.nan)
    return close / prev - 1


def cumulative_returns(close: pd.Series) -> pd.Series:
    """Return since the first available close (0.10 means +10 %)."""
    close = pd.Series(close, dtype="float64")
    valid = close.dropna()
    if valid.empty or valid.iloc[0] <= 0:
        return close * np.nan
    return close / valid.iloc[0] - 1


def moving_average(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average; NaN until ``window`` observations exist."""
    if window < 1:
        raise ValueError("window must be at least 1")
    return pd.Series(close, dtype="float64").rolling(window, min_periods=window).mean()


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        # No losses: RSI is 100 if there were gains, 50 if the price never moved.
        return 50.0 if avg_gain == 0 else 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    The first value appears after ``period`` price changes (i.e. on observation
    number ``period + 1``). Earlier rows are NaN.
    """
    close = pd.Series(close, dtype="float64")
    result = pd.Series(np.nan, index=close.index, dtype="float64")
    valid = close.dropna()
    if period < 1 or len(valid) <= period:
        return result

    values = valid.to_numpy()
    delta = np.diff(values)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    rsi = np.full(len(values), np.nan)
    rsi[period] = _rsi_from_averages(avg_gain, avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rsi[i + 1] = _rsi_from_averages(avg_gain, avg_loss)

    result.loc[valid.index] = rsi
    return result


def historical_volatility(
    close: pd.Series, window: int = 20, annualize: bool = True
) -> pd.Series:
    """Rolling standard deviation of daily returns (optionally annualized).

    Returned as a fraction: 0.25 means 25 %. Annualizing multiplies by
    sqrt(252). NaN until ``window`` returns are available.
    """
    rets = daily_returns(close)
    vol = rets.rolling(window, min_periods=window).std(ddof=1)
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def drawdown_series(close: pd.Series) -> pd.Series:
    """Percentage fall from the running peak (always <= 0)."""
    close = pd.Series(close, dtype="float64")
    running_peak = close.cummax()
    return close / running_peak.replace(0, np.nan) - 1


def max_drawdown(close: pd.Series):
    """Return (max_drawdown, peak_label, trough_label) for a close series.

    The drawdown is negative (-0.25 = a 25 % fall from peak to trough).
    Labels are the series index values (dates when indexed by Date).
    """
    close = pd.Series(close, dtype="float64").dropna()
    if close.empty:
        return np.nan, None, None
    dd = drawdown_series(close)
    trough = dd.idxmin()
    peak = close.loc[:trough].idxmax()
    return float(dd.loc[trough]), peak, trough


def normalize_to_base(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase each column so that the first common date equals ``base``.

    Only dates where every column has a price are kept, so all series start
    on the same day. Returns an empty frame if there are no common dates.
    """
    common = prices.dropna(how="any")
    if common.empty:
        return common
    first = common.iloc[0].replace(0, np.nan)
    return common / first * base


def add_indicators(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 50,
    rsi_period: int = 14,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Add Return, MA_<short>, MA_<long>, RSI, Volatility and Drawdown columns.

    ``df`` must contain ONE ticker. Indicators are computed on the full history
    passed in, so slice to a date window *after* calling this function.
    """
    out = df.sort_values("Date").reset_index(drop=True).copy()
    close = out["Close"]
    out["Return"] = daily_returns(close)
    out[f"MA_{short}"] = moving_average(close, short)
    out[f"MA_{long}"] = moving_average(close, long)
    out["RSI"] = compute_rsi(close, rsi_period)
    out["Volatility"] = historical_volatility(close, vol_window, annualize=True)
    out["Drawdown"] = drawdown_series(close)
    return out


# --------------------------------------------------------------------------
# KPI, watchlist and portfolio helpers
# --------------------------------------------------------------------------


def kpi_summary(df: pd.DataFrame, max_gap_days: int = 5) -> Optional[dict]:
    """Headline numbers for ONE ticker over the rows supplied.

    The daily change compares the last two observations. If they are more
    than ``max_gap_days`` calendar days apart (a data gap, not a normal
    weekend or holiday) the daily change is reported as None.
    """
    if df is None or df.empty:
        return None
    df = df.sort_values("Date")
    latest = df.iloc[-1]

    daily_change = daily_pct = prev_close = None
    if len(df) >= 2:
        prev = df.iloc[-2]
        gap = (latest["Date"] - prev["Date"]).days
        if gap <= max_gap_days and prev["Close"] > 0:
            prev_close = float(prev["Close"])
            daily_change = float(latest["Close"] - prev["Close"])
            daily_pct = daily_change / prev_close

    period_return = None
    first_close = df.iloc[0]["Close"]
    if len(df) >= 2 and first_close > 0:
        period_return = float(latest["Close"] / first_close - 1)

    return {
        "latest_date": latest["Date"],
        "latest_close": float(latest["Close"]),
        "prev_close": prev_close,
        "daily_change": daily_change,
        "daily_pct": daily_pct,
        "period_return": period_return,
        "period_high": float(df["High"].max()),
        "period_low": float(df["Low"].min()),
        "avg_volume": float(df["Volume"].mean()),
        "observations": int(len(df)),
    }


def portfolio_summary(holdings: pd.DataFrame, latest_prices: dict):
    """Value hypothetical holdings.

    ``holdings`` needs Ticker, Shares and Purchase_Price columns.
    ``latest_prices`` maps ticker -> latest close (or None if unavailable).

    Returns (table, totals). Holdings without a price are marked
    "Price unavailable" and excluded from market-value totals; rows with
    unusable inputs are marked "Invalid input".
    """
    columns = [
        "Ticker", "Shares", "Purchase_Price", "Cost", "Latest_Close",
        "Market_Value", "Unrealized_PL", "Unrealized_Return", "Allocation", "Status",
    ]
    rows = []
    for _, r in holdings.iterrows():
        raw_ticker = r.get("Ticker")
        ticker_blank = raw_ticker is None or pd.isna(raw_ticker) or str(raw_ticker).strip() == ""
        if ticker_blank and pd.isna(r.get("Shares")) and pd.isna(r.get("Purchase_Price")):
            continue  # completely empty row
        symbol = normalize_ticker(raw_ticker)
        shares = pd.to_numeric(r.get("Shares"), errors="coerce")
        buy = pd.to_numeric(r.get("Purchase_Price"), errors="coerce")
        row = dict.fromkeys(columns, np.nan)
        row.update(Ticker=symbol or str(raw_ticker), Shares=shares, Purchase_Price=buy)

        if symbol is None or pd.isna(shares) or pd.isna(buy) or shares <= 0 or buy <= 0:
            row["Status"] = "Invalid input"
            rows.append(row)
            continue

        row["Cost"] = float(shares * buy)
        price = latest_prices.get(symbol)
        if price is None or pd.isna(price):
            row["Status"] = "Price unavailable"
        else:
            row["Latest_Close"] = float(price)
            row["Market_Value"] = float(shares * price)
            row["Unrealized_PL"] = row["Market_Value"] - row["Cost"]
            row["Unrealized_Return"] = row["Unrealized_PL"] / row["Cost"]
            row["Status"] = "OK"
        rows.append(row)

    table = pd.DataFrame(rows, columns=columns)
    priced = table["Status"] == "OK"
    total_mv = float(table.loc[priced, "Market_Value"].sum())
    priced_cost = float(table.loc[priced, "Cost"].sum())
    if total_mv > 0:
        table.loc[priced, "Allocation"] = table.loc[priced, "Market_Value"] / total_mv

    totals = {
        "total_cost": float(table["Cost"].sum()),  # all valid holdings
        "priced_cost": priced_cost,  # holdings that have a price
        "market_value": total_mv,
        "unrealized_pl": total_mv - priced_cost if priced.any() else None,
        "unrealized_return": (total_mv / priced_cost - 1) if priced_cost > 0 else None,
        "unpriced_count": int((table["Status"] == "Price unavailable").sum()),
        "invalid_count": int((table["Status"] == "Invalid input").sum()),
    }
    return table, totals

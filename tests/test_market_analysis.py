"""Unit tests for the sample data and src/market_analysis.py.

Run from the project root (no extra packages needed):

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import market_analysis as ma  # noqa: E402

CSV_PATH = ROOT / "data" / "sample_stock_prices.csv"


class SampleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        cls.df = ma.load_sample_data(CSV_PATH)

    def test_file_exists_and_has_expected_columns(self):
        self.assertTrue(CSV_PATH.exists())
        self.assertEqual(list(self.raw.columns), ma.REQUIRED_COLUMNS)

    def test_ticker_count_and_history_length(self):
        n_tickers = self.raw["Ticker"].nunique()
        self.assertTrue(8 <= n_tickers <= 12)
        span_days = (self.raw["Date"].max() - self.raw["Date"].min()).days
        self.assertGreaterEqual(span_days, 365)

    def test_no_data_quality_issues(self):
        self.assertEqual(ma.validate_price_data(self.raw), [])

    def test_ohlc_relationships(self):
        d = self.raw
        self.assertTrue((d["High"] >= d[["Open", "Close", "Low"]].max(axis=1)).all())
        self.assertTrue((d["Low"] <= d[["Open", "Close", "High"]].min(axis=1)).all())

    def test_prices_positive_and_volume_nonnegative(self):
        self.assertTrue((self.raw[ma.PRICE_COLUMNS] > 0).all().all())
        self.assertTrue((self.raw["Volume"] >= 0).all())

    def test_dates_sorted_per_ticker_without_duplicates(self):
        for _, s in self.raw.groupby("Ticker")["Date"]:
            self.assertTrue(s.is_monotonic_increasing)
        self.assertFalse(self.raw.duplicated(["Ticker", "Date"]).any())

    def test_adjusted_close_not_above_close(self):
        self.assertTrue((self.raw["Adjusted_Close"] <= self.raw["Close"] + 1e-9).all())

    def test_generator_is_reproducible(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import generate_sample_data as gen

        regenerated = gen.generate()
        regenerated["Date"] = pd.to_datetime(regenerated["Date"])
        pd.testing.assert_frame_equal(
            regenerated.reset_index(drop=True),
            self.raw.sort_values(["Ticker", "Date"]).reset_index(drop=True),
            check_dtype=False,
        )

    def test_validator_detects_problems(self):
        bad = self.raw.copy()
        bad.loc[0, "High"] = bad.loc[0, "Low"] - 1
        bad.loc[1, "Close"] = -5
        issues = " ".join(ma.validate_price_data(bad))
        self.assertIn("High is below", issues)
        self.assertIn("non-positive", issues)


class IndicatorTests(unittest.TestCase):
    def test_daily_returns(self):
        r = ma.daily_returns(pd.Series([100.0, 110.0, 99.0]))
        self.assertTrue(np.isnan(r.iloc[0]))
        self.assertAlmostEqual(r.iloc[1], 0.10)
        self.assertAlmostEqual(r.iloc[2], -0.10)

    def test_daily_returns_zero_denominator(self):
        r = ma.daily_returns(pd.Series([0.0, 5.0]))
        self.assertTrue(np.isnan(r.iloc[1]))

    def test_moving_average_values_and_warmup(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        m = ma.moving_average(s, 3)
        self.assertTrue(m.iloc[:2].isna().all())
        self.assertEqual(m.iloc[2], 2.0)
        self.assertEqual(m.iloc[4], 4.0)

    def test_rsi_insufficient_history_is_nan(self):
        rsi = ma.compute_rsi(pd.Series(np.arange(1.0, 11.0)), 14)
        self.assertTrue(rsi.isna().all())

    def test_rsi_extremes_and_range(self):
        up = ma.compute_rsi(pd.Series(np.arange(1.0, 40.0)), 14)
        self.assertEqual(up.iloc[14], 100.0)
        flat = ma.compute_rsi(pd.Series([10.0] * 30), 14)
        self.assertEqual(flat.iloc[20], 50.0)
        down = ma.compute_rsi(pd.Series(np.arange(40.0, 1.0, -1.0)), 14)
        self.assertEqual(down.iloc[20], 0.0)
        rng = np.random.default_rng(1)
        noisy = ma.compute_rsi(pd.Series(100 + rng.normal(0, 1, 200).cumsum()), 14).dropna()
        self.assertTrue(((noisy >= 0) & (noisy <= 100)).all())

    def test_rsi_matches_hand_calculation(self):
        # Wilder's classic textbook series (first RSI value, 14 changes)
        prices = pd.Series([
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84,
            46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
        ])
        rsi = ma.compute_rsi(prices, 14)
        delta = prices.diff().dropna()
        expected_rs = delta.clip(lower=0).mean() / (-delta.clip(upper=0)).mean()
        self.assertAlmostEqual(rsi.iloc[14], 100 - 100 / (1 + expected_rs), places=6)
        self.assertAlmostEqual(rsi.iloc[14], 70.46, places=1)

    def test_volatility(self):
        close = pd.Series(100 * np.cumprod(1 + np.tile([0.01, -0.01], 15)))
        vol = ma.historical_volatility(close, 20, annualize=False)
        self.assertTrue(vol.iloc[:20].isna().all())
        expected = ma.daily_returns(close).iloc[1:21].std(ddof=1)
        self.assertAlmostEqual(vol.iloc[20], expected)
        annual = ma.historical_volatility(close, 20, annualize=True)
        self.assertAlmostEqual(annual.iloc[20], expected * np.sqrt(252))

    def test_max_drawdown(self):
        idx = pd.date_range("2025-01-01", periods=6, freq="D")
        close = pd.Series([100, 120, 90, 110, 80, 100], index=idx, dtype=float)
        mdd, peak, trough = ma.max_drawdown(close)
        self.assertAlmostEqual(mdd, 80 / 120 - 1)
        self.assertEqual(peak, idx[1])
        self.assertEqual(trough, idx[4])
        self.assertTrue(np.isnan(ma.max_drawdown(pd.Series([], dtype=float))[0]))
        self.assertTrue((ma.drawdown_series(close) <= 0).all())

    def test_cumulative_returns(self):
        c = ma.cumulative_returns(pd.Series([50.0, 55.0, 25.0]))
        self.assertEqual(list(c.round(6)), [0.0, 0.1, -0.5])

    def test_normalization_starts_at_100_on_common_dates(self):
        idx = pd.date_range("2025-01-01", periods=4, freq="D")
        wide = pd.DataFrame(
            {"A": [10.0, 11.0, 12.0, 15.0], "B": [np.nan, 200.0, 220.0, 180.0]}, index=idx
        )
        norm = ma.normalize_to_base(wide)
        self.assertEqual(norm.index[0], idx[1])  # first date where both exist
        self.assertTrue((norm.iloc[0] == 100.0).all())
        self.assertAlmostEqual(norm["A"].iloc[-1], 15 / 11 * 100)
        self.assertAlmostEqual(norm["B"].iloc[-1], 90.0)
        self.assertTrue(ma.normalize_to_base(wide.iloc[:1]).empty)

    def test_add_indicators_on_sample(self):
        df = ma.load_sample_data(CSV_PATH)
        one = ma.add_indicators(df[df["Ticker"] == "NVLX"])
        for col in ["Return", "MA_20", "MA_50", "RSI", "Volatility", "Drawdown"]:
            self.assertIn(col, one.columns)
        self.assertTrue(one["MA_50"].iloc[:49].isna().all())
        self.assertTrue(one["MA_50"].iloc[49:].notna().all())
        self.assertTrue(one["RSI"].dropna().between(0, 100).all())
        self.assertTrue((one["Volatility"].dropna() > 0).all())


class KpiPortfolioTests(unittest.TestCase):
    def _frame(self, dates, closes):
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(dates),
                "Close": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Volume": [1000, 2000, 3000][: len(closes)],
            }
        )

    def test_kpis(self):
        k = ma.kpi_summary(self._frame(["2025-01-02", "2025-01-03", "2025-01-06"], [100.0, 102.0, 99.96]))
        self.assertAlmostEqual(k["daily_change"], -2.04)
        self.assertAlmostEqual(k["daily_pct"], -0.02)
        self.assertAlmostEqual(k["period_return"], -0.0004)
        self.assertEqual(k["period_high"], 103.0)
        self.assertEqual(k["period_low"], 98.96)
        self.assertEqual(k["avg_volume"], 2000.0)

    def test_kpi_no_daily_change_across_data_gap_or_single_row(self):
        gap = ma.kpi_summary(self._frame(["2025-01-02", "2025-02-20"], [100.0, 120.0]))
        self.assertIsNone(gap["daily_change"])
        self.assertAlmostEqual(gap["period_return"], 0.2)
        single = ma.kpi_summary(self._frame(["2025-01-02"], [100.0]))
        self.assertIsNone(single["daily_change"])
        self.assertIsNone(single["period_return"])
        self.assertIsNone(ma.kpi_summary(self._frame([], [])))

    def test_portfolio(self):
        holdings = pd.DataFrame(
            {
                "Ticker": ["aaa", "BBB", "CCC", "bad ticker!", None],
                "Shares": [10, 5, 4, 3, np.nan],
                "Purchase_Price": [10.0, 20.0, 5.0, 1.0, np.nan],
            }
        )
        table, totals = ma.portfolio_summary(holdings, {"AAA": 15.0, "BBB": 10.0, "CCC": None})
        by = table.set_index("Ticker")
        self.assertEqual(len(table), 4)  # blank row dropped
        self.assertAlmostEqual(by.loc["AAA", "Cost"], 100.0)
        self.assertAlmostEqual(by.loc["AAA", "Market_Value"], 150.0)
        self.assertAlmostEqual(by.loc["AAA", "Unrealized_PL"], 50.0)
        self.assertAlmostEqual(by.loc["AAA", "Unrealized_Return"], 0.5)
        self.assertAlmostEqual(by.loc["BBB", "Unrealized_PL"], -50.0)
        self.assertEqual(by.loc["CCC", "Status"], "Price unavailable")
        self.assertTrue(np.isnan(by.loc["CCC", "Market_Value"]))
        self.assertEqual(table["Status"].tolist().count("Invalid input"), 1)
        self.assertAlmostEqual(by.loc["AAA", "Allocation"], 0.75)
        self.assertAlmostEqual(by.loc["BBB", "Allocation"], 0.25)
        self.assertAlmostEqual(totals["market_value"], 200.0)
        self.assertAlmostEqual(totals["priced_cost"], 200.0)
        self.assertAlmostEqual(totals["total_cost"], 220.0)
        self.assertAlmostEqual(totals["unrealized_pl"], 0.0)
        self.assertEqual(totals["unpriced_count"], 1)
        self.assertEqual(totals["invalid_count"], 1)

    def test_portfolio_with_no_prices(self):
        holdings = pd.DataFrame({"Ticker": ["ZZZ"], "Shares": [1], "Purchase_Price": [2.0]})
        _, totals = ma.portfolio_summary(holdings, {})
        self.assertIsNone(totals["unrealized_pl"])
        self.assertIsNone(totals["unrealized_return"])


class LiveDataTests(unittest.TestCase):
    """Live-data handling is tested with injected fetchers, so no network is used."""

    @staticmethod
    def _fake_history(symbol, period):
        idx = pd.date_range("2025-03-03", periods=5, freq="B", tz="America/New_York")
        hist = pd.DataFrame(
            {
                "Open": [10, 11, np.nan, 12, 13],
                "High": [11, 12, 12, 13, 14],
                "Low": [9, 10, 10, 11, 12],
                "Close": [10.5, 11.5, np.nan, 12.5, 13.5],
                "Adj Close": [10.4, 11.4, np.nan, 12.4, 13.4],
                "Volume": [100, 200, 0, 300, np.nan],
            },
            index=idx,
        )
        return hist, "Test Corp"

    def test_successful_fetch_is_standardized(self):
        res = ma.fetch_live_data("test", "1mo", fetcher=self._fake_history)
        self.assertIsNone(res.error)
        self.assertEqual(res.company, "Test Corp")
        self.assertIsNotNone(res.fetched_at)
        self.assertEqual(list(res.data.columns), ma.REQUIRED_COLUMNS)
        self.assertEqual(len(res.data), 4)  # row with missing Close dropped
        self.assertIsNone(res.data["Date"].dt.tz)
        self.assertEqual(res.data["Ticker"].iloc[0], "TEST")
        self.assertEqual(ma.validate_price_data(res.data), [])

    def test_multiindex_columns_are_handled(self):
        def fetcher(symbol, period):
            hist, name = self._fake_history(symbol, period)
            hist.columns = pd.MultiIndex.from_product([hist.columns, [symbol]])
            return hist, name

        res = ma.fetch_live_data("AAPL", "1y", fetcher=fetcher)
        self.assertIsNone(res.error)
        self.assertEqual(len(res.data), 4)

    def test_network_failure_returns_error_not_exception(self):
        def boom(symbol, period):
            raise ConnectionError("offline")

        res = ma.fetch_live_data("AAPL", "1y", fetcher=boom)
        self.assertIsNone(res.data)
        self.assertIn("Could not retrieve", res.error)

    def test_rate_limit_style_error(self):
        def limited(symbol, period):
            raise RuntimeError("Too Many Requests")

        res = ma.fetch_live_data("AAPL", "1y", fetcher=limited)
        self.assertIsNone(res.data)
        self.assertIn("rate limit", res.error)

    def test_empty_result_and_missing_package(self):
        empty = ma.fetch_live_data("ZZZZ", "1y", fetcher=lambda s, p: (pd.DataFrame(), None))
        self.assertIsNone(empty.data)
        self.assertIn("No data", empty.error)

        def no_pkg(symbol, period):
            raise ImportError("no yfinance")

        self.assertIn("not installed", ma.fetch_live_data("AAPL", "1y", fetcher=no_pkg).error)

    def test_invalid_inputs(self):
        self.assertIsNone(ma.fetch_live_data("bad symbol!", "1y").data)
        self.assertIsNone(ma.fetch_live_data("AAPL", "10 years").data)
        self.assertEqual(ma.normalize_ticker(" brk-b "), "BRK-B")
        self.assertIsNone(ma.normalize_ticker(""))
        self.assertIsNone(ma.normalize_ticker(None))


if __name__ == "__main__":
    unittest.main()

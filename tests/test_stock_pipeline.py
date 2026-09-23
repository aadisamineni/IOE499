"""Synthetic fixtures only; these never become downloaded dataset rows."""
import unittest

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from NvidiaDatapull import FEATURES, derive_features, sessions, validate, validate_hourly


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = sessions("2025-08-01", "2025-10-08")
        n = len(self.index)
        close = np.arange(n, dtype=float) + 100
        self.frame = pd.DataFrame({
            "ticker": "NVDA", "open": close - 1, "high": close + 2,
            "low": close - 2, "close": close, "adjusted_close": close * 0.9,
            "volume": np.arange(n) + 1000, "dividends": 0.0, "stock_splits": 0.0,
        }, index=self.index)
        self.frame.index.name = "trading_date"

    def test_feature_math_and_warmup_lag(self):
        output = derive_features(self.frame)
        pos = self.index.get_loc(pd.Timestamp("2025-10-01"))
        row = output.iloc[pos]
        prior_closes = self.frame.adjusted_close.iloc[pos-20:pos+1].to_numpy()
        returns = prior_closes[1:] / prior_closes[:-1] - 1
        self.assertAlmostEqual(row[FEATURES[3]], np.std(returns, ddof=1))
        self.assertAlmostEqual(row[FEATURES[4]], self.frame.volume.iloc[pos-19:pos+1].mean())
        self.assertAlmostEqual(row[FEATURES[1]], row.close / row.open - 1)
        self.assertAlmostEqual(row[FEATURES[2]],
                               row.adjusted_close / self.frame.adjusted_close.iloc[pos-5] - 1)
        for feature in FEATURES:
            self.assertEqual(row[feature + "_lag1_session"], output.iloc[pos-1][feature])
        self.assertTrue(pd.isna(output.iloc[0][FEATURES[0]]))
        self.assertTrue(pd.isna(output.iloc[19][FEATURES[3]]))
        self.assertFalse(output.loc["2025-10-01":].isna().any().any())

    def test_missing_session_and_bad_prices_detected(self):
        bad = self.frame.drop(self.index[5]).copy()
        bad.iloc[0, bad.columns.get_loc("high")] = 1
        bad.iloc[1, bad.columns.get_loc("volume")] = np.nan
        report = validate(bad, "2025-08-01", "2025-10-08")
        self.assertFalse(report["passed"])
        self.assertEqual(report["missing_sessions"], [str(self.index[5].date())])
        self.assertFalse(report["checks"]["valid_ohlc_bounds"])
        self.assertFalse(report["checks"]["no_missing_source_values"])

    def test_five_percent_inclusive_thresholds_and_missing_history(self):
        frame = self.frame.iloc[:9].copy()
        frame["adjusted_close"] = [100, 105, 100, 95, 100, 104.99, 100, 94.99, np.nan]
        result = derive_features(frame)
        expected = pd.Series([np.nan, 1, 0, -1, 1, 0, 0, -1, np.nan],
                             index=frame.index, name="price_move_5pct")
        pd.testing.assert_series_equal(result.price_move_5pct, expected)
        pd.testing.assert_series_equal(result.price_move_5pct_lag1_session,
                                       expected.shift(1), check_names=False)

    def test_one_percent_inclusive_thresholds_and_missing_history(self):
        frame = self.frame.iloc[:9].copy()
        frame["adjusted_close"] = [100, 101, 100, 99, 100, 100.99, 100, 98.99, np.nan]
        result = derive_features(frame)
        expected = pd.Series([np.nan, 1, 0, -1, 1, 0, 0, -1, np.nan],
                             index=frame.index, name="price_move_1pct")
        pd.testing.assert_series_equal(result.price_move_1pct, expected)
        pd.testing.assert_series_equal(result.price_move_1pct_lag1_session,
                                       expected.shift(1), check_names=False)

    def test_actions_retained_and_no_weekend_rows(self):
        self.frame.loc[self.index[2], "dividends"] = 0.01
        report = validate(self.frame, "2025-08-01", "2025-10-08")
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["corporate_actions_for_review"]), 1)
        self.assertTrue((self.index.dayofweek < 5).all())
        self.assertNotIn(pd.Timestamp("2025-09-01"), self.index)

    def test_hourly_validation_handles_normal_and_early_close_sessions(self):
        schedule = mcal.get_calendar("NASDAQ").schedule(
            start_date="2025-11-26", end_date="2025-11-28")
        stamps = []
        for row in schedule.itertuples():
            count = int(np.ceil((row.market_close - row.market_open).total_seconds() / 3600))
            stamps.extend(row.market_open + pd.to_timedelta(np.arange(count), unit="h"))
        index = pd.DatetimeIndex(stamps).tz_convert("America/New_York")
        close = np.arange(len(index), dtype=float) + 100
        hourly = pd.DataFrame({
            "ticker": "NVDA", "open": close, "high": close + 2,
            "low": close - 2, "close": close + 1, "adjusted_close": close + 1,
            "volume": np.arange(len(index)) + 1000,
            "dividends": 0.0, "stock_splits": 0.0,
        }, index=index)
        hourly.index.name = "trading_timestamp"

        report = validate_hourly(hourly, "2025-11-26", "2025-11-28")
        self.assertTrue(report["passed"])
        self.assertEqual(report["observed_session_count"], 2)

        missing_bar = validate_hourly(hourly.drop(hourly.index[-1]),
                                      "2025-11-26", "2025-11-28")
        self.assertFalse(missing_bar["passed"])
        self.assertFalse(missing_bar["checks"]["expected_bar_count_each_session"])
        self.assertEqual(missing_bar["bar_count_review"][0]["actual_bar_count"], 3)

        provider_gap = hourly.copy()
        provider_gap.loc[provider_gap.index[-1],
                         ["open", "high", "low", "close", "adjusted_close"]] = np.nan
        provider_gap.loc[provider_gap.index[-1], "volume"] = 0
        gap_report = validate_hourly(provider_gap, "2025-11-26", "2025-11-28")
        self.assertFalse(gap_report["passed"])
        self.assertTrue(gap_report["structural_validation_passed"])
        self.assertFalse(gap_report["complete_price_data"])
        self.assertEqual(gap_report["missing_price_bar_count"], 1)


if __name__ == "__main__":
    unittest.main()

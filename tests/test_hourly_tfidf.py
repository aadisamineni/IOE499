"""Deterministic tests for the leakage-safe hourly TF-IDF dataset."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PATH = Path(__file__).resolve().parents[1] / "tf-idf-hourly/hourly_tfidf.py"
SPEC = spec_from_file_location("hourly_tfidf", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HourlyTfidfTests(unittest.TestCase):
    def bars(self):
        rows = []
        prices = [100, 101, 100.5, 101.5, 101, 102, 102.5,
                  103, 102, 103, 104, 103.5, 104.5, 105]
        starts = []
        for date in ("2025-12-01", "2025-12-02"):
            starts.extend(pd.date_range(
                f"{date} 09:30", periods=7, freq="h", tz="America/New_York"))
        for index, (start, opening) in enumerate(zip(starts, prices)):
            close = opening * (1.002 if index % 2 == 0 else 0.997)
            rows.append({
                "trading_timestamp": start.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "ticker": "NVDA",
                "open": opening,
                "high": max(opening, close) + 0.2,
                "low": min(opening, close) - 0.2,
                "close": close,
                "adjusted_close": close,
                "volume": 1000 + index,
            })
        return pd.DataFrame(rows)

    def news(self):
        base = {
            "summary": "Chip update",
            "relevance_type": "direct",
            "excluded": "false",
            "event_representative": "true",
        }
        return pd.DataFrame([
            {
                **base, "article_id": "a", "headline": "Before cutoff",
                "published_at_utc": "2025-12-01T16:00:00Z",
                "duplicate_group_id": "group-a",
            },
            {
                **base, "article_id": "b", "headline": "Exactly at cutoff",
                "published_at_utc": "2025-12-01T16:30:00Z",
                "duplicate_group_id": "group-b",
            },
            {
                **base, "article_id": "c", "headline": "Excluded article",
                "published_at_utc": "2025-12-01T17:00:00Z",
                "duplicate_group_id": "group-c", "excluded": "true",
            },
            {
                **base, "article_id": "d", "headline": "Syndicated copy",
                "published_at_utc": "2025-12-01T17:05:00Z",
                "duplicate_group_id": "group-d", "event_representative": "false",
            },
            {
                **base, "article_id": "e", "headline": "Overnight update",
                "published_at_utc": "2025-12-02T03:00:00Z",
                "duplicate_group_id": "group-e",
            },
        ])

    def test_news_is_strictly_before_cutoff_and_used_once(self):
        examples, audit = MODULE.build_hourly_examples(
            self.bars(), self.news(), {"2025-12-01", "2025-12-02"},
            volatility_window=2)
        by_start = examples.set_index(examples.bar_start_new_york.dt.strftime("%Y-%m-%d %H:%M"))
        eleven_thirty = by_start.loc["2025-12-01 11:30", "document"]
        twelve_thirty = by_start.loc["2025-12-01 12:30", "document"]
        next_open = by_start.loc["2025-12-02 09:30", "document"]
        self.assertIn("Before cutoff", eleven_thirty)
        self.assertNotIn("Exactly at cutoff", eleven_thirty)
        self.assertIn("Exactly at cutoff", twelve_thirty)
        self.assertIn("Overnight update", next_open)
        self.assertNotIn("Excluded article", "\n".join(examples.document))
        self.assertNotIn("Syndicated copy", "\n".join(examples.document))
        self.assertEqual("\n".join(examples.document).count("Before cutoff"), 1)
        self.assertEqual(audit["partial_final_bars_excluded"], 2)
        self.assertTrue(examples.bar_duration_minutes.eq(60).all())

    def test_incomplete_utc_coverage_removes_overnight_window(self):
        examples, audit = MODULE.build_hourly_examples(
            self.bars(), self.news(), {"2025-12-01"}, volatility_window=2)
        self.assertNotIn("2025-12-02", set(examples.trading_date))
        self.assertGreater(audit["bars_excluded_for_incomplete_news_coverage"], 0)

    def test_exactly_flat_bar_is_not_forced_into_a_direction(self):
        bars = self.bars()
        bars.loc[2, "close"] = bars.loc[2, "open"]
        bars.loc[2, "adjusted_close"] = bars.loc[2, "open"]
        examples, audit = MODULE.build_hourly_examples(
            bars, self.news(), {"2025-12-01", "2025-12-02"}, volatility_window=2)
        starts = set(examples.bar_start_new_york.dt.strftime("%Y-%m-%d %H:%M"))
        self.assertNotIn("2025-12-01 11:30", starts)
        self.assertEqual(audit["zero_return_bars_excluded"], 1)

    def test_date_split_never_divides_a_session(self):
        frame = pd.DataFrame({
            "trading_date": np.repeat(
                [f"2025-12-{day:02d}" for day in range(1, 11)], 3),
            "value": np.arange(30),
        })
        train, test = MODULE.split_by_trading_date(frame, 0.7)
        self.assertEqual(train.trading_date.nunique(), 7)
        self.assertEqual(test.trading_date.nunique(), 3)
        self.assertFalse(set(train.trading_date) & set(test.trading_date))

    def test_class_prior_uses_only_training_labels(self):
        probabilities = MODULE.class_prior_probabilities(
            np.array([-1, -1, 1]), 2, np.array([-1, 1]))
        np.testing.assert_allclose(probabilities[0], [2 / 3, 1 / 3])
        np.testing.assert_allclose(probabilities[0], probabilities[1])

    def test_vectorizer_removes_bare_numbers_but_keeps_mixed_chip_names(self):
        vectorizer = MODULE.make_vectorizer(min_df=1, max_df=1.0, max_features=20)
        vectorizer.fit(["Nvidia H100 in 2026", "New chips in 2025"])
        terms = set(vectorizer.get_feature_names_out())
        self.assertIn("h100", terms)
        self.assertIn("nvidia", terms)
        self.assertNotIn("2026", terms)
        self.assertNotIn("2025", terms)


if __name__ == "__main__":
    unittest.main()

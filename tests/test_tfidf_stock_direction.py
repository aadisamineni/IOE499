"""Small deterministic tests for the TF-IDF session-building logic."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


PATH = Path(__file__).resolve().parents[1] / "tf-idf/tfidf_stock_direction.py"
SPEC = spec_from_file_location("tfidf_stock_direction", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TfidfStockDirectionTests(unittest.TestCase):
    def daily(self):
        return pd.DataFrame({
            "trading_date": ["2025-12-02", "2025-12-03", "2025-12-04"],
            "matched_news_date": ["2025-12-01", "2025-12-02", "2025-12-03"],
            "price_move_2pct": [1, 0, -1],
            "adjusted_close_return_1d": [0.03, 0.0, -0.025],
            "adjusted_close_return_1d_lag1_session": [0.01, 0.03, 0.0],
            "daily_return_volatility_20d_lag1_session": [0.02, 0.02, 0.021],
            "within_configured_utc_collection_range": [1, 1, 0],
        })

    def matches(self):
        return pd.DataFrame([
            {
                "trading_date": "2025-12-02", "matched_news_date": "2025-12-01",
                "article_match_key": "a", "headline": "Nvidia launches chip",
                "summary": "New product", "event_representative": "true",
                "publication_timestamp_utc": "2025-12-01T12:00:00+00:00",
                "duplicate_group_id": "group-a",
            },
            {
                "trading_date": "2025-12-02", "matched_news_date": "2025-12-01",
                "article_match_key": "b", "headline": "Syndicated copy",
                "summary": "Duplicate", "event_representative": "false",
                "publication_timestamp_utc": "2025-12-01T13:00:00+00:00",
                "duplicate_group_id": "group-a",
            },
            {
                "trading_date": "2025-12-04", "matched_news_date": "2025-12-03",
                "article_match_key": "c", "headline": "Outside coverage",
                "summary": "Excluded session", "event_representative": "true",
                "publication_timestamp_utc": "2025-12-03T12:00:00+00:00",
                "duplicate_group_id": "group-c",
            },
        ])

    def test_documents_keep_sessions_and_only_representatives(self):
        sessions = MODULE.build_session_documents(self.daily(), self.matches())
        self.assertEqual(sessions.trading_date.tolist(), ["2025-12-02", "2025-12-03"])
        self.assertEqual(sessions.iloc[0].representative_article_count, 1)
        self.assertIn("Nvidia launches chip", sessions.iloc[0].document)
        self.assertNotIn("Syndicated copy", sessions.iloc[0].document)
        self.assertEqual(sessions.iloc[1].representative_article_count, 0)
        self.assertEqual(sessions.iloc[1].document, "")
        self.assertAlmostEqual(sessions.iloc[0].log1p_article_count, np.log(2))

    def test_prior_calendar_date_is_enforced(self):
        daily = self.daily()
        daily.loc[0, "matched_news_date"] = "2025-11-30"
        with self.assertRaisesRegex(ValueError, "previous calendar date"):
            MODULE.build_session_documents(daily, self.matches())

    def test_class_prior_probabilities_follow_training_distribution(self):
        probabilities = MODULE.class_prior_probabilities(
            np.array([-1, 0, 0, 1]), n_rows=2)
        np.testing.assert_allclose(probabilities[0], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(probabilities[0], probabilities[1])


if __name__ == "__main__":
    unittest.main()

"""Tests for exact previous-New-York-calendar-day news matching."""

import unittest

import pandas as pd

from matched_one_day.match_previous_day import (
    build_matches,
    classify_move,
    prepare_articles,
    prepare_stock,
    validate_results,
)


class OneDayMatchingTests(unittest.TestCase):
    def test_two_percent_feature_has_inclusive_symmetric_thresholds(self):
        cases = {
            -0.0201: -1,
            -0.02: -1,
            -0.0199: 0,
            0.0: 0,
            0.0199: 0,
            0.02: 1,
            0.0201: 1,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(classify_move(value), expected)

    def test_december_second_matches_december_first_new_york_news(self):
        stock_input = pd.DataFrame({
            "trading_date": ["2025-12-02"],
            "ticker": ["NVDA"],
            "adjusted_close_return_1d": ["0.025"],
        })
        news_input = pd.DataFrame([
            {
                "article_id": "late-utc",
                "headline": "Still December 1 in New York",
                "published_at_utc": "2025-12-02T03:30:00Z",
                "relevance_type": "direct",
            },
            {
                "article_id": "december-2",
                "headline": "December 2 in New York",
                "published_at_utc": "2025-12-02T15:00:00Z",
                "relevance_type": "direct",
            },
        ])
        stock = prepare_stock(stock_input, None, None, 0.02)
        articles, review, duplicates = prepare_articles(news_input)
        collection = {"start_date": "2025-10-01", "end_date": "2026-05-01"}
        summary, matches = build_matches(stock, articles, collection)

        self.assertTrue(review.empty)
        self.assertEqual(duplicates, 0)
        self.assertEqual(stock.iloc[0].matched_news_date, "2025-12-01")
        self.assertEqual(matches.article_id.tolist(), ["late-utc"])
        self.assertEqual(matches.iloc[0].news_date_new_york, "2025-12-01")
        self.assertEqual(summary.iloc[0].price_move_2pct, 1)
        self.assertEqual(summary.iloc[0].matched_article_count, 1)
        self.assertTrue(validate_results(stock, summary, matches, 0.02))

    def test_zero_news_stock_session_is_preserved(self):
        stock_input = pd.DataFrame({
            "trading_date": ["2025-12-08"],
            "ticker": ["NVDA"],
            "adjusted_close_return_1d": ["-0.005"],
        })
        news_input = pd.DataFrame(columns=[
            "article_id", "headline", "published_at_utc", "relevance_type",
        ])
        stock = prepare_stock(stock_input, None, None, 0.02)
        articles, _, _ = prepare_articles(news_input)
        collection = {"start_date": "2025-10-01", "end_date": "2026-05-01"}
        summary, matches = build_matches(stock, articles, collection)

        self.assertTrue(matches.empty)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary.iloc[0].matched_article_count, 0)
        self.assertEqual(summary.iloc[0].price_move_2pct, 0)
        self.assertTrue(validate_results(stock, summary, matches, 0.02))


if __name__ == "__main__":
    unittest.main()

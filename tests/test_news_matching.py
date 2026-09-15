"""Temporal edge cases using synthetic fixtures; no fixture enters real outputs."""
import unittest

import pandas as pd

from matching.match_news import (calendar_events, coverage_table, membership,
    parse_timestamp, prepare_articles, revision_exclusion, window_bounds, match_events)


class NewsMatchingTests(unittest.TestCase):
    def test_timestamp_timezone_and_date_only_rejection(self):
        for value in ["", "2025-10-01", "2025-10-01T12:30:00", "nonsense"]:
            parsed, problem = parse_timestamp(value)
            self.assertIsNone(parsed)
            self.assertTrue(problem)
        parsed, problem = parse_timestamp("2025-10-01T12:30:00-04:00")
        self.assertFalse(problem)
        self.assertEqual(parsed, pd.Timestamp("2025-10-01T16:30:00Z"))

    def test_exclusive_cutoff_inclusive_window_edges(self):
        cutoff = pd.Timestamp("2025-10-10T20:00:00Z")
        self.assertIsNone(membership(cutoff, cutoff))
        self.assertIsNone(membership(cutoff + pd.Timedelta(seconds=1), cutoff))
        for hours, flags in [(24, (1, 1, 1)), (24.001, (0, 1, 1)),
                             (72, (0, 1, 1)), (72.001, (0, 0, 1)), (168, (0, 0, 1))]:
            result = membership(cutoff - pd.Timedelta(hours=hours), cutoff)
            self.assertEqual(tuple(result[k] for k in ["in_24h", "in_72h", "in_7d"]), flags)
        self.assertIsNone(membership(cutoff - pd.Timedelta(hours=168, seconds=1), cutoff))

    def test_seven_calendar_days_across_dst(self):
        spring = pd.Timestamp("2026-03-09T20:00:00Z")
        fall = pd.Timestamp("2025-11-03T21:00:00Z")
        self.assertEqual((spring - window_bounds(spring)["7d"]).total_seconds() / 3600, 167)
        self.assertEqual((fall - window_bounds(fall)["7d"]).total_seconds() / 3600, 169)

    def stock(self, dates):
        return pd.DataFrame({"trading_date": dates, "ticker": "NVDA",
                             "price_move_1pct": "1", "price_move_5pct": "1"})

    def test_previous_session_holiday_early_close_and_overlap(self):
        events, stock, _, _ = calendar_events(self.stock(["2025-11-28", "2025-12-01"]),
                                              "2025-11-28", "2025-12-01")
        self.assertEqual(events[0]["cutoff_utc"], "2025-11-26T21:00:00+00:00")
        self.assertEqual(events[1]["cutoff_utc"], "2025-11-28T18:00:00+00:00")
        self.assertEqual(events[1]["cutoff_new_york"], "2025-11-28T13:00:00-05:00")
        self.assertEqual(events[0]["price_move_1pct"], "1")
        self.assertEqual(events[0]["shift_1pct_up"], events[0]["shift_5pct_up"])

    def article(self, **overrides):
        return {"article_id": "a", "headline": "Nvidia article", "summary": "text",
                "source": "Example", "url": "https://example.com/a", "event_category": "existing",
                "published_at_utc": "2025-10-06T19:00:00Z",
                "retrieved_at_utc": "2026-09-14T12:00:00Z",
                "relevance_type": "direct", **overrides}

    def test_exact_copies_distinct_coverage_revisions_and_ambiguity(self):
        row = self.article()
        distinct = self.article(article_id="b", url="https://example.com/b")
        revised = self.article(article_id="c", updated_at_utc="2025-10-06T21:00:00Z")
        ambiguous = self.article(article_id="d", published_at_utc="2025-10-06")
        frame = pd.DataFrame([row, row, distinct, revised, ambiguous]).fillna("")
        articles, review, exclusions = prepare_articles(frame, {})
        self.assertEqual(len(articles), 3)
        self.assertEqual(len(review), 1)
        self.assertEqual(len(exclusions), 2)
        cutoff = pd.Timestamp("2025-10-06T20:00:00Z")
        self.assertEqual(revision_exclusion(articles[0], cutoff), "")
        self.assertIn("known_post_cutoff_revision", revision_exclusion(articles[2], cutoff))

    def test_multiple_events_and_zero_matches_with_coverage_gap(self):
        dates = ["2025-10-07", "2025-10-08", "2025-10-20"]
        events, _, _, _ = calendar_events(self.stock(dates), dates[0], dates[-1])
        articles, _, _ = prepare_articles(pd.DataFrame([self.article()]), {})
        config = {"tickers": ["NVDA"], "start_date": "2025-10-01", "end_date": "2025-10-10"}
        coverage = coverage_table([], dates[0], dates[-1], config)
        summary, matches, excluded = match_events(events, articles, coverage, config)
        self.assertEqual(len(summary), 3)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches.article_match_key.nunique(), 1)
        self.assertEqual(summary.iloc[-1].article_count_7d, 0)
        self.assertEqual(summary.iloc[-1].known_collection_gap, 1)
        self.assertEqual(matches.historical_availability_uncertain.sum(), 2)


if __name__ == "__main__":
    unittest.main()

"""Synthetic unit tests for clustering and labeling. These do not touch source data."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_events import label_event_clusters  # noqa: E402
from news_clusters import build_fixed_clusters  # noqa: E402


UTC = "UTC"


class FixedClusterTests(unittest.TestCase):
    def test_window_is_anchored_and_does_not_chain(self) -> None:
        published = pd.to_datetime(
            ["2026-01-05T15:00:00Z", "2026-01-05T15:04:00Z", "2026-01-05T15:08:00Z"]
        )
        articles = pd.DataFrame(
            {
                "article_id": [1, 2, 3],
                "published_at_utc": published,
                "headline": ["one", "two", "three"],
                "summary": ["", "", ""],
                "source": ["source"] * 3,
                "relevance_type": ["direct"] * 3,
                "event_category": ["test"] * 3,
                "duplicate_group_id": ["a", "b", "c"],
                "article_text": ["one", "two", "three"],
                "session_date": ["2026-01-05"] * 3,
                "market_open": [pd.Timestamp("2026-01-05T14:30:00Z")] * 3,
                "market_close": [pd.Timestamp("2026-01-05T21:00:00Z")] * 3,
            }
        )

        clusters, members = build_fixed_clusters(
            articles, cluster_minutes=5, latency_minutes=1
        )

        self.assertEqual(len(clusters), 2)
        self.assertEqual(clusters.article_count.tolist(), [2, 1])
        self.assertEqual(len(members), 3)
        self.assertEqual(
            clusters.loc[0, "decision_time_utc"],
            pd.Timestamp("2026-01-05T15:06:00Z"),
        )


class LabelTests(unittest.TestCase):
    @staticmethod
    def minute_quotes() -> pd.DataFrame:
        minute_ends = pd.date_range(
            "2026-01-05T14:01:00Z", "2026-01-05T16:30:00Z", freq="min"
        )
        midpoint = 100.0 * np.exp(np.arange(len(minute_ends)) * 0.001)
        return pd.DataFrame(
            {
                "minute_end_utc": minute_ends,
                "nvda_midpoint": midpoint,
                "nvda_bid_price": midpoint - 0.005,
                "nvda_ask_price": midpoint + 0.005,
                "nvda_bid_size": 100.0,
                "nvda_ask_size": 100.0,
                "nvda_spread_bps": 0.01 / midpoint * 10_000.0,
                "nvda_last_quote_utc": minute_ends - pd.Timedelta(seconds=1),
                "nvda_valid_quote_updates": 10.0,
            }
        )

    def test_labels_and_news_contamination_are_horizon_specific(self) -> None:
        clusters = pd.DataFrame(
            {
                "news_cluster_id": ["first", "second"],
                "session_date": ["2026-01-05", "2026-01-05"],
                "market_open_utc": pd.to_datetime(
                    ["2026-01-05T14:00:00Z"] * 2
                ),
                "market_close_utc": pd.to_datetime(
                    ["2026-01-05T21:00:00Z"] * 2
                ),
                "cluster_start_utc": pd.to_datetime(
                    ["2026-01-05T14:30:00Z", "2026-01-05T14:50:00Z"]
                ),
                "cluster_window_end_utc": pd.to_datetime(
                    ["2026-01-05T14:35:00Z", "2026-01-05T14:55:00Z"]
                ),
                "decision_time_utc": pd.to_datetime(
                    ["2026-01-05T14:36:00Z", "2026-01-05T14:56:00Z"]
                ),
                "combined_text": ["first event", "second event"],
            }
        )

        labeled = label_event_clusters(
            clusters=clusters,
            primary_quotes=self.minute_quotes(),
            primary_symbol="NVDA",
            horizons=[5, 15],
            pre_vol_minutes=20,
            minimum_pre_vol_observations=10,
            z_cutoff=1.0,
            spread_multiple=1.0,
            max_quote_staleness_seconds=30.0,
        )

        first = labeled.iloc[0]
        self.assertEqual(first.label_5m, "up")
        self.assertTrue(first.clean_5m)
        self.assertFalse(first.clean_15m)
        self.assertEqual(first.new_clusters_during_15m, 1)
        self.assertGreater(first.max_label_return_5m, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_dataset import (  # noqa: E402
    MarketLookup,
    SessionMapper,
    add_news_contamination,
    assign_chronological_split,
    build_events,
    label_event,
    make_schedule,
)
from train_models import (  # noqa: E402
    TaskSpec,
    development_test_split,
    feature_groups,
    make_model,
)


class SessionMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapper = SessionMapper(
            make_schedule(
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-12T00:00:00Z"),
            )
        )

    def test_market_periods_and_next_session(self) -> None:
        premarket = self.mapper.context(pd.Timestamp("2026-01-05T13:00:00Z"))
        regular = self.mapper.context(pd.Timestamp("2026-01-05T15:00:00Z"))
        afterhours = self.mapper.context(pd.Timestamp("2026-01-05T22:00:00Z"))
        weekend = self.mapper.context(pd.Timestamp("2026-01-10T17:00:00Z"))

        self.assertEqual(premarket.market_period, "premarket")
        self.assertEqual(premarket.outcome_session_date, "2026-01-05")
        self.assertEqual(regular.market_period, "regular")
        self.assertEqual(afterhours.market_period, "afterhours")
        self.assertEqual(afterhours.outcome_session_date, "2026-01-06")
        self.assertEqual(weekend.market_period, "overnight")
        self.assertEqual(weekend.outcome_session_date, "2026-01-12")

    def test_overnight_articles_are_batched(self) -> None:
        context = self.mapper.context(pd.Timestamp("2026-01-10T17:00:00Z"))
        rows = []
        for index, timestamp in enumerate(
            pd.to_datetime(["2026-01-10T17:00:00Z", "2026-01-11T18:00:00Z"])
        ):
            rows.append(
                {
                    "article_id": index,
                    "published_at_utc": timestamp,
                    "headline": f"headline {index}",
                    "summary": "summary",
                    "combined_text": f"headline {index}. summary",
                    "source": "source",
                    "event_category": "test",
                    "relevance_type": "direct",
                    "duplicate_group_size": 1,
                    **context.__dict__,
                }
            )
        events = build_events(pd.DataFrame(rows))
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].article_count, 2)
        self.assertEqual(events.iloc[0].market_period, "overnight")


class EventLabelTests(unittest.TestCase):
    @staticmethod
    def market_frame() -> pd.DataFrame:
        starts = pd.date_range("2026-01-05T14:00:00Z", periods=181, freq="min")
        increments = 0.0001 + np.sin(np.arange(len(starts)) / 3.0) * 0.00005
        close = 100 * np.exp(np.cumsum(increments))
        open_price = np.r_[100.0, close[:-1]]
        return pd.DataFrame(
            {
                "bar_start_utc": starts,
                "minute_end_utc": starts + pd.Timedelta(minutes=1),
                "open": open_price,
                "high": close * 1.0001,
                "low": open_price * 0.9999,
                "close": close,
                "volume": 1_000 + np.arange(len(starts)),
                "trade_count": 100 + np.arange(len(starts)),
                "vwap": (open_price + close) / 2,
                "price": close,
            }
        )

    @staticmethod
    def event() -> pd.Series:
        return pd.Series(
            {
                "event_id": "event-1",
                "published_at_utc": pd.Timestamp("2026-01-05T15:00:30Z"),
                "analysis_time_utc": pd.Timestamp("2026-01-05T15:00:30Z"),
                "market_period": "regular",
                "event_session_date": "2026-01-05",
                "outcome_session_date": "2026-01-05",
                "market_open_utc": pd.Timestamp("2026-01-05T14:30:00Z"),
                "market_close_utc": pd.Timestamp("2026-01-05T21:00:00Z"),
                "outcome_open_utc": pd.Timestamp("2026-01-05T14:30:00Z"),
                "outcome_close_utc": pd.Timestamp("2026-01-05T21:00:00Z"),
                "previous_close_utc": pd.Timestamp("2026-01-02T21:00:00Z"),
                "combined_text": "Nvidia announces a new product",
            }
        )

    def test_reaction_and_continuation_have_distinct_anchors(self) -> None:
        labeled = label_event(
            self.event(),
            MarketLookup(self.market_frame()),
            horizons=[1, 5, 15],
            decision_delays=[1, 5],
            z_cutoff=1.0,
            spread_multiple=1.0,
        )
        self.assertTrue(labeled["market_data_available"])
        self.assertGreater(labeled["reaction_return_5m"], 0)
        self.assertGreater(labeled["continuation_return_1m_to_15m"], 0)
        expected = (
            labeled["reaction_return_15m"] - labeled["initial_return_1m"]
        )
        self.assertAlmostEqual(
            labeled["continuation_return_1m_to_15m"], expected, places=12
        )

    def test_contamination_is_diagnostic(self) -> None:
        labeled = pd.DataFrame(
            [
                label_event(
                    self.event(),
                    MarketLookup(self.market_frame()),
                    horizons=[5],
                    decision_delays=[1],
                    z_cutoff=1.0,
                    spread_multiple=1.0,
                )
            ]
        )
        result = add_news_contamination(
            labeled,
            pd.Series(
                pd.to_datetime(
                    ["2026-01-05T15:00:30Z", "2026-01-05T15:03:00Z"]
                )
            ),
            [5],
        )
        self.assertEqual(result.iloc[0].later_news_count_5m, 1)
        self.assertFalse(result.iloc[0].clean_5m)

    def test_overnight_reaction_includes_gap_but_continuation_does_not(self) -> None:
        friday_starts = pd.date_range(
            "2026-01-02T20:29:00Z", periods=31, freq="min"
        )
        monday_pre_starts = pd.date_range(
            "2026-01-05T14:00:00Z", periods=30, freq="min"
        )
        monday_rth_starts = pd.date_range(
            "2026-01-05T14:30:00Z", periods=61, freq="min"
        )
        starts = friday_starts.append(monday_pre_starts).append(monday_rth_starts)
        friday_close = np.linspace(99.0, 100.0, len(friday_starts))
        pre_close = np.linspace(108.0, 109.0, len(monday_pre_starts))
        rth_close = np.linspace(110.1, 116.1, len(monday_rth_starts))
        close = np.r_[friday_close, pre_close, rth_close]
        open_price = np.r_[friday_close, pre_close, np.r_[110.0, rth_close[:-1]]]
        market = pd.DataFrame(
            {
                "bar_start_utc": starts,
                "minute_end_utc": starts + pd.Timedelta(minutes=1),
                "open": open_price,
                "high": np.maximum(open_price, close) * 1.0001,
                "low": np.minimum(open_price, close) * 0.9999,
                "close": close,
                "volume": 1_000.0,
                "trade_count": 100.0,
                "vwap": (open_price + close) / 2,
                "price": close,
            }
        )
        event = self.event().copy()
        event["published_at_utc"] = pd.Timestamp("2026-01-04T18:00:00Z")
        event["analysis_time_utc"] = pd.Timestamp("2026-01-05T14:30:00Z")
        event["market_period"] = "overnight"
        event["previous_close_utc"] = pd.Timestamp("2026-01-02T21:00:00Z")
        event["outcome_open_utc"] = pd.Timestamp("2026-01-05T14:30:00Z")
        labeled = label_event(
            event,
            MarketLookup(market),
            horizons=[1, 5, 15],
            decision_delays=[1, 5],
            z_cutoff=1.0,
            spread_multiple=1.0,
        )
        self.assertGreater(
            labeled["reaction_return_15m"], labeled["post_anchor_return_15m"]
        )
        self.assertAlmostEqual(
            labeled["reaction_return_15m"],
            labeled["total_return_including_gap_15m"],
        )
        self.assertAlmostEqual(
            labeled["continuation_return_5m_to_15m"],
            labeled["post_anchor_return_15m"] - labeled["initial_return_5m"],
        )


class ModelTests(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        rows = []
        for index in range(60):
            label = index % 2
            rows.append(
                {
                    "event_id": f"event-{index}",
                    "outcome_session_date": str(
                        (pd.Timestamp("2025-10-01") + pd.Timedelta(days=index)).date()
                    ),
                    "combined_text": (
                        "strong demand guidance raised"
                        if label
                        else "weak demand guidance reduced"
                    ),
                    "impact_label_5m": label,
                    "prior_return_1m": 0.001 if label else -0.001,
                    "pre_volatility_30m": 0.002,
                    "market_period": "regular",
                    "event_category": "guidance",
                    "source": "wire",
                    "relevance_type": "direct",
                    "article_count": 1,
                    "novelty_30d": 0.5,
                }
            )
        return pd.DataFrame(rows)

    def test_date_split_and_combined_model(self) -> None:
        frame = self.frame()
        development, test = development_test_split(frame, 0.15)
        self.assertLess(
            development.outcome_session_date.max(), test.outcome_session_date.min()
        )
        task = TaskSpec("impact_5m", "classification", "impact_label_5m")
        groups = feature_groups(frame, task)
        model = make_model(task, "market_news_tfidf", groups, 1.0, 100, 8)
        model.fit(development, development.impact_label_5m)
        probabilities = model.predict_proba(test)[:, 1]
        self.assertEqual(len(probabilities), len(test))
        self.assertTrue(np.isfinite(probabilities).all())

    def test_split_keeps_dates_together(self) -> None:
        frame = self.frame()
        frame = pd.concat([frame, frame.iloc[[0]].assign(event_id="duplicate-date")])
        split = assign_chronological_split(frame)
        counts = split.groupby("outcome_session_date").chronological_split.nunique()
        self.assertTrue(counts.eq(1).all())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import sys
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common import load_config, normalize_url  # noqa: E402
from generate_report import coverage_periods  # noqa: E402
from process_news import classify_article, event_group, exact_deduplicate  # noqa: E402


def base_article(**overrides):
    article = {
        "article_id": "1",
        "queried_tickers": "NVDA",
        "headline": "",
        "summary": "",
        "source": "Example",
        "url": "https://example.com/story",
        "normalized_url": "https://example.com/story",
        "published_at_utc": "2025-10-01T12:00:00Z",
        "retrieved_at_utc": "2026-09-14T12:00:00Z",
        "finnhub_related": "",
        "finnhub_category": "company",
        "image_url": "",
        "raw_observation_count": 1,
        "exact_duplicate_group_id": "exact_test",
    }
    article.update(overrides)
    return article


class CommonTests(unittest.TestCase):
    def test_project_config_and_dates(self):
        config = load_config()
        self.assertEqual(config["collection"]["start_date"], "2025-10-01")
        self.assertEqual(config["collection"]["end_date"], "2026-05-01")
        self.assertEqual(config["collection"]["primary_ticker"], "NVDA")
        self.assertEqual(len(config["collection"]["tickers"]), 9)

    def test_url_normalization_removes_tracking(self):
        value = "HTTPS://Example.COM/a/story/?utm_source=x&b=2&a=1#section"
        self.assertEqual(normalize_url(value), "https://example.com/a/story?a=1&b=2")


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_direct_nvidia_classification(self):
        article = base_article(
            headline="Nvidia unveils a new Blackwell GPU",
            summary="The new system targets data centers.",
            finnhub_related="NVDA",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "direct")
        self.assertEqual(result["event_category"], "product_announcement")
        self.assertFalse(result["excluded"])

    def test_indirect_hbm_classification_has_reason(self):
        article = base_article(
            queried_tickers="MU",
            headline="Micron expands HBM capacity for AI accelerators",
            summary="High bandwidth memory supply will increase next year.",
            finnhub_related="MU",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "indirect")
        self.assertEqual(result["event_category"], "manufacturing_packaging_hbm")
        self.assertIn("Nvidia", result["relevance_reason"])
        self.assertFalse(result["excluded"])

    def test_generic_semiconductor_article_is_not_retained(self):
        article = base_article(
            queried_tickers="TSM",
            headline="Semiconductor sector employment increases",
            summary="Chip companies hired more accountants.",
            finnhub_related="TSM",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "not_relevant")
        self.assertTrue(result["excluded"])
        self.assertIn("insufficient_nvidia_connection", result["exclusion_reason"])

    def test_earnings_are_excluded_separately_from_relevance(self):
        article = base_article(
            headline="Nvidia earnings recap: revenue beats expectations",
            summary="Nvidia reported quarterly results.",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "direct")
        self.assertTrue(result["excluded"])
        self.assertIn("earnings_report_preview_or_recap", result["exclusion_reason"])

    def test_hyperscaler_data_center_needs_ai_and_spending_context(self):
        article = base_article(
            queried_tickers="META",
            headline="Solar company supports Meta data center with power project",
            summary="Meta will acquire renewable electricity from the project.",
            finnhub_related="META",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "not_relevant")
        self.assertTrue(result["excluded"])

    def test_generic_recommendation_language_is_excluded(self):
        article = base_article(
            headline="Nvidia raises its investment and so should you",
            summary="Nvidia is discussed.",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "direct")
        self.assertIn("generic_recommendation_or_listicle", result["exclusion_reason"])

    def test_generic_semiconductor_export_story_needs_ai_chip_link(self):
        article = base_article(
            queried_tickers="MU",
            headline="Semiconductor stocks fall after export control update",
            summary="A wafer equipment company expects a revenue headwind.",
            finnhub_related="MU",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "not_relevant")

    def test_generic_cloud_partnership_is_not_hardware_relevance(self):
        article = base_article(
            queried_tickers="MSFT",
            headline="Microsoft partners with a university on datacenter community programs",
            summary="The pledge concerns education near a data center.",
            finnhub_related="MSFT",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "not_relevant")

    def test_cloud_cybersecurity_antitrust_is_not_nvidia_regulation(self):
        article = base_article(
            queried_tickers="GOOGL",
            headline="EU antitrust regulators review Google's Wiz deal",
            summary="The acquisition concerns cloud cybersecurity software.",
            finnhub_related="GOOGL",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "not_relevant")

    def test_summary_only_nvidia_mention_needs_material_event(self):
        article = base_article(
            queried_tickers="MSFT",
            headline="Broad market trades near unchanged",
            summary="Microsoft and Nvidia were among many stocks mentioned.",
            finnhub_related="MSFT,NVDA",
        )
        result = classify_article(article, self.config)
        self.assertEqual(result["relevance_type"], "direct")
        self.assertIn(
            "summary_only_mention_without_material_nvidia_event",
            result["exclusion_reason"],
        )


class DeduplicationTests(unittest.TestCase):
    def test_id_and_normalized_url_deduplication_unions_tickers(self):
        observations = [
            {
                "article_id": "100",
                "queried_ticker": "NVDA",
                "headline": "Nvidia announcement",
                "summary": "Short",
                "source": "Wire",
                "url": "https://example.com/a?utm_source=x",
                "normalized_url": "https://example.com/a",
                "published_at_utc": "2025-10-01T12:00:00Z",
                "retrieved_at_utc": "2026-09-14T12:00:00Z",
                "finnhub_related": "NVDA",
                "finnhub_category": "company",
                "image_url": "",
            },
            {
                "article_id": "100",
                "queried_ticker": "MSFT",
                "headline": "Nvidia announcement",
                "summary": "A longer summary about the same announcement.",
                "source": "Wire",
                "url": "https://example.com/a",
                "normalized_url": "https://example.com/a",
                "published_at_utc": "2025-10-01T12:00:00Z",
                "retrieved_at_utc": "2026-09-14T12:01:00Z",
                "finnhub_related": "NVDA,MSFT",
                "finnhub_category": "company",
                "image_url": "",
            },
        ]
        result = exact_deduplicate(observations)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["queried_tickers"], "MSFT|NVDA")
        self.assertEqual(result[0]["raw_observation_count"], 2)
        self.assertIn("MSFT", result[0]["finnhub_related"])

    def test_similar_headlines_share_event_group(self):
        config = load_config()
        first = classify_article(
            base_article(
                article_id="1",
                headline="Nvidia unveils new Blackwell GPU platform",
                summary="Nvidia announced the platform for data centers.",
            ),
            config,
        )
        second = classify_article(
            base_article(
                article_id="2",
                headline="Nvidia unveils its new Blackwell GPU platform",
                summary="Nvidia introduced the platform.",
                published_at_utc="2025-10-01T13:00:00Z",
            ),
            config,
        )
        articles = [first, second]
        event_group(articles, config)
        self.assertEqual(articles[0]["duplicate_group_id"], articles[1]["duplicate_group_id"])
        self.assertEqual(sum(article["event_representative"] for article in articles), 1)
        self.assertEqual(sum(article["excluded"] for article in articles), 1)


class CoverageTests(unittest.TestCase):
    def test_missing_and_empty_windows_are_reported(self):
        windows = [
            {
                "ticker": "NVDA",
                "from_date": "2025-10-01",
                "to_date": "2025-10-01",
                "status": "success",
            },
            {
                "ticker": "NVDA",
                "from_date": "2025-10-02",
                "to_date": "2025-10-02",
                "status": "success_empty_unverified",
            },
        ]
        periods, complete = coverage_periods(
            windows, ["NVDA"], date(2025, 10, 1), date(2025, 10, 3)
        )
        self.assertFalse(complete)
        self.assertEqual([row["coverage_state"] for row in periods], [
            "empty_response_unverified",
            "missing_no_terminal_request",
        ])


if __name__ == "__main__":
    unittest.main()

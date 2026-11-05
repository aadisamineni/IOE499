import csv
import importlib.util
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("merge_daily_news", Path(__file__).with_name("merge_daily_news.py"))
news = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(news)


class MergeTest(unittest.TestCase):
    def test_merges_symbols_and_providers_and_retains_empty_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            providers = {name: root / name for name in ("alpaca", "finnhub")}
            for provider, folder in providers.items():
                folder.mkdir()
                for symbol in news.SYMBOLS:
                    with (folder / f"{symbol}_news.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=("article_id", "published_at_utc", "headline", "retrieved_at_utc"))
                        writer.writeheader()
                        if symbol in ("NVDA", "SPY"):
                            writer.writerow({"article_id": "1", "published_at_utc": "2025-05-01T23:30:00-01:00",
                                             "headline": provider, "retrieved_at_utc": "2025-05-03T00:00:00Z"})
            out = root / "output"
            with patch.object(news, "PROVIDERS", providers):
                summary = news.merge(date(2025, 5, 1), date(2025, 5, 2), 499, out)
            with (out / "daily_weights.csv").open() as handle:
                days = list(csv.DictReader(handle))
            with (out / "news_with_daily_weights.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["article_count"] for row in days], ["0", "2"])
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["article_id"], "")
            self.assertEqual({row["provider"] for row in rows[1:]}, {"alpaca", "finnhub"})
            self.assertTrue(all(row["query_symbols"] == "NVDA,SPY" for row in rows[1:]))
            self.assertEqual(summary["days_without_articles"], 1)
            for day in days:
                self.assertEqual(sum(Decimal(day[column]) for column in news.WEIGHT_COLUMNS), Decimal(1))
            for row in rows[1:]:
                self.assertEqual([row[column] for column in news.WEIGHT_COLUMNS],
                                 [days[1][column] for column in news.WEIGHT_COLUMNS])
            self.assertEqual(news.weights_for_day(date(2025, 5, 2), 499), news.weights_for_day(date(2025, 5, 2), 499))


if __name__ == "__main__":
    unittest.main()

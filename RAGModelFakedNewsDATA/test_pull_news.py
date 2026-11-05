import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("rag_news", Path(__file__).with_name("pull_news.py"))
news = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(news)


def snapshot(base, symbol, start, end, status, articles, retrieved):
    raw = base / "raw"
    path = raw / f"{symbol}_{start}.json"
    raw.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"retrieved_at_utc": retrieved, "response": articles}))
    row = {"record_type": "request_window", "window_id": f"{symbol}_{start}_{end}",
           "ticker": symbol, "from_date": start, "to_date": end, "status": status,
           "raw_file": str(path.relative_to(base))}
    with (raw / "request_manifest.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")


class ExportTest(unittest.TestCase):
    def test_exports_unique_records_and_truthful_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, legacy = Path(tmp) / "new", Path(tmp) / "legacy"
            article = {"id": 123, "datetime": 1746057600, "headline": "old"}
            snapshot(legacy, "NVDA", "2025-05-01", "2025-05-01", "success", [article], "2025-05-03T00:00:00Z")
            snapshot(root, "NVDA", "2025-05-01", "2025-05-01", "success", [{**article, "headline": "new"}], "2025-05-04T00:00:00Z")
            snapshot(root, "SPY", "2025-05-01", "2025-05-01", "success_empty_unverified", [], "2025-05-04T00:00:00Z")
            with patch.object(news, "ROOT", root), patch.object(news, "LEGACY", legacy):
                news.export(date(2025, 5, 2))
            with (root / "processed/NVDA_news.csv").open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["headline"], "new")
            self.assertEqual(records[0]["collection_origin"], "current_pull")
            with (root / "processed/daily_coverage.csv").open() as handle:
                coverage = {(row["query_symbol"], row["date_utc"]): row for row in csv.DictReader(handle)}
            self.assertEqual(coverage[("SPY", "2025-05-01")]["current_pull_status"], "success_empty_unverified")
            self.assertEqual(coverage[("SPY", "2025-05-02")]["current_pull_status"], "not_requested")
            self.assertEqual(coverage[("NVDA", "2025-05-01")]["existing_pull_status"], "success")


if __name__ == "__main__":
    unittest.main()

import csv
import importlib.util
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("alpaca_news", Path(__file__).with_name("pull_alpaca_news.py"))
news = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(news)


class AlpacaExportTest(unittest.TestCase):
    def test_export_deduplicates_by_id_and_keeps_symbol_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "raw" / "2025-05"
            folder.mkdir(parents=True)
            first = {"id": 1, "created_at": "2025-05-01T12:00:00Z", "symbols": ["NVDA", "SPY"],
                     "headline": "old", "content": "<p>body</p>"}
            second = {**first, "headline": "updated"}
            for index, article in enumerate((first, second), 1):
                (folder / f"page_{index:05}.json").write_text(json.dumps({"retrieved_at_utc": f"2025-05-0{index}T00:00:00Z",
                    "response": {"news": [article], "next_page_token": None}}))
            with patch.object(news, "ROOT", root):
                news.export(date(2025, 5, 31))
            for symbol in ("NVDA", "SPY"):
                with (root / f"processed/{symbol}_news.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["headline"], "updated")
                self.assertEqual(rows[0]["content_html"], "<p>body</p>")
            with (root / "processed/PFE_news.csv").open() as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()

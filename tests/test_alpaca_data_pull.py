"""Offline tests for the resumable Alpaca downloader helpers."""

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "alpaca data pull/pull_alpaca_nvda.py"
SPEC = spec_from_file_location("pull_alpaca_nvda", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AlpacaDataPullTests(unittest.TestCase):
    def test_labeled_credentials_are_parsed_without_label(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.env"
            path.write_text("Private Key = 'secret-value'\n", encoding="utf-8")
            value = MODULE.read_labeled_secret(path, {"privatekey"})
            self.assertEqual(value, "secret-value")

    def test_daily_intervals_are_inclusive_and_nonoverlapping(self):
        intervals = list(MODULE.daily_intervals(
            date(2025, 10, 1), date(2025, 10, 2)))
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0][1], "2025-10-01T00:00:00Z")
        self.assertEqual(intervals[0][2], "2025-10-01T23:59:59.999999999Z")
        self.assertEqual(intervals[1][1], "2025-10-02T00:00:00Z")

    def test_nested_symbol_and_action_payloads_are_preserved(self):
        quotes = MODULE.normalize_records({"NVDA": [{"t": "one"}]}, "NVDA")
        self.assertEqual(quotes, [{"t": "one"}])
        actions = MODULE.normalize_records({
            "cash_dividends": [{"id": "a"}],
            "splits": [{"id": "b"}],
        }, "NVDA")
        self.assertEqual(actions[0]["_action_type"], "cash_dividends")
        self.assertEqual(actions[1]["_action_type"], "splits")

    def test_query_fingerprint_changes_with_feed(self):
        first = MODULE.query_fingerprint(
            "trades", "NVDA", "sip", "start", "end", {})
        second = MODULE.query_fingerprint(
            "trades", "NVDA", "iex", "start", "end", {})
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

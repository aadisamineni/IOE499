#!/usr/bin/env python3
"""Collect Finnhub company-news snapshots and export one CSV per symbol."""

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / "news data pull"
sys.path.insert(0, str(LEGACY / "scripts"))
import collect_news as collector_module
from common import read_jsonl

SYMBOLS = ("NVDA", "SPY", "META", "PFE")
START = date(2025, 5, 1)
FIELDS = ("article_id", "query_symbol", "published_at_utc", "headline", "summary",
          "source", "url", "related", "category", "image_url", "retrieved_at_utc",
          "collection_origin")


class Collector(collector_module.FinnhubCollector):
    def _request_once(self, ticker, start, end):
        try:
            return super()._request_once(ticker, start, end)
        except OSError:
            return collector_module.ApiResponse(None, None, {},
                datetime.now(timezone.utc).isoformat(), "network_or_timeout")


def existing_coverage():
    latest = {}
    for item in read_jsonl(LEGACY / "raw" / "request_manifest.jsonl"):
        if item.get("record_type") == "request_window" and item.get("window_id"):
            latest[item["window_id"]] = item
    covered = set()
    for item in latest.values():
        if item.get("status") != "success" or item.get("ticker") not in SYMBOLS:
            continue
        path = LEGACY / str(item.get("raw_file") or "")
        if not path.is_file():
            continue
        day = date.fromisoformat(item["from_date"])
        end = date.fromisoformat(item["to_date"])
        while day <= end:
            covered.add((item["ticker"], day))
            day += timedelta(days=1)
    return covered


def credential(path):
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key and path:
        key = Path(path).expanduser().read_text(encoding="utf-8").strip()
        if "=" in key:
            key = key.split("=", 1)[1].strip()
        key = key.strip("\"'")
    if not key:
        raise SystemExit("Set FINNHUB_API_KEY or supply --key-file.")
    return key


def export(end):
    rows = {symbol: {} for symbol in SYMBOLS}
    stats = {symbol: {"raw_files": 0, "invalid_timestamps": 0} for symbol in SYMBOLS}
    coverage = {}
    for origin, base in (("current_pull", ROOT), ("existing_pull", LEGACY)):
        latest = {}
        for item in read_jsonl(base / "raw" / "request_manifest.jsonl"):
            if item.get("record_type") == "request_window" and item.get("window_id"):
                latest[item["window_id"]] = item
        for item in latest.values():
            symbol = item.get("ticker")
            if symbol not in rows or item.get("status") == "split_potential_truncation":
                continue
            try:
                day = max(START, date.fromisoformat(item["from_date"]))
                last_day = min(end, date.fromisoformat(item["to_date"]))
            except (KeyError, TypeError, ValueError):
                continue
            while day <= last_day:
                coverage[(symbol, str(day), origin)] = item.get("status", "unknown")
                day += timedelta(days=1)
            if item.get("status") not in collector_module.TERMINAL_SUCCESS_STATUSES:
                continue
            path = base / str(item.get("raw_file") or "")
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("response"), list):
                continue
            stats[symbol]["raw_files"] += 1
            for article in payload["response"]:
                if not isinstance(article, dict):
                    continue
                try:
                    timestamp = datetime.fromtimestamp(int(article["datetime"]), timezone.utc)
                except (KeyError, ValueError, TypeError, OverflowError, OSError):
                    stats[symbol]["invalid_timestamps"] += 1
                    continue
                if not START <= timestamp.date() <= end:
                    continue
                article_id = str(article.get("id") or "")
                url = str(article.get("url") or "")
                headline = str(article.get("headline") or "")
                key = article_id or url or f"{timestamp.isoformat()}|{headline}"
                row = {"article_id": article_id, "query_symbol": symbol,
                       "published_at_utc": timestamp.isoformat().replace("+00:00", "Z"),
                       "headline": headline, "summary": article.get("summary", ""),
                       "source": article.get("source", ""), "url": url,
                       "related": article.get("related", ""), "category": article.get("category", ""),
                       "image_url": article.get("image", ""),
                       "retrieved_at_utc": payload.get("retrieved_at_utc", ""),
                       "collection_origin": origin}
                # Prefer the newest snapshot, while retaining older saved-only articles.
                prior = rows[symbol].get(key)
                if prior is None or row["retrieved_at_utc"] > prior["retrieved_at_utc"]:
                    rows[symbol][key] = row
    output = ROOT / "processed"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "daily_coverage.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("query_symbol", "date_utc", "current_pull_status", "existing_pull_status"))
        for symbol in SYMBOLS:
            day = START
            while day <= end:
                writer.writerow((symbol, str(day), coverage.get((symbol, str(day), "current_pull"), "not_requested"),
                                 coverage.get((symbol, str(day), "existing_pull"), "not_available")))
                day += timedelta(days=1)
    for symbol in SYMBOLS:
        records = sorted(rows[symbol].values(), key=lambda r: (r["published_at_utc"], r["article_id"]))
        path = output / f"{symbol}_news.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
        stats[symbol]["article_count"] = len(records)
        stats[symbol]["first_publication_utc"] = records[0]["published_at_utc"] if records else None
        stats[symbol]["last_publication_utc"] = records[-1]["published_at_utc"] if records else None
        print(f"{symbol}: exported {len(records)} articles", flush=True)
    (output / "export_summary.json").write_text(json.dumps({"requested_start_utc": str(START),
        "requested_end_utc": str(end), "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": stats}, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", help="Optional local credential file (do not commit it)")
    parser.add_argument("--end", type=date.fromisoformat, default=datetime.now(timezone.utc).date())
    parser.add_argument("--export-only", action="store_true", help="Export available snapshots without network access")
    args = parser.parse_args()
    if args.end < START:
        parser.error("--end must be on or after 2025-05-01")
    if not args.export_only:
        config = json.loads((LEGACY / "config.json").read_text(encoding="utf-8"))
        config["collection"].update(start_date=str(START), end_date=str(args.end),
                                    tickers=list(SYMBOLS), primary_ticker="NVDA")
        collector_module.PROJECT_ROOT = ROOT
        collector_module.MANIFEST_PATH = ROOT / "raw" / "request_manifest.jsonl"
        collector = Collector(config, credential(args.key_file))
        already_saved = existing_coverage()
        # A previous request for today's still-in-progress window must be refreshed.
        if args.end == datetime.now(timezone.utc).date():
            for window_id, item in list(collector.latest_manifest.items()):
                if item.get("from_date", "") <= str(args.end) <= item.get("to_date", ""):
                    del collector.latest_manifest[window_id]
        try:
            # Start with recent dates for all four symbols, then work backward.
            windows = list(collector_module.date_windows(START, args.end, collector.initial_window_days))
            for index, (start, end) in enumerate(reversed(windows), 1):
                for symbol in SYMBOLS:
                    if all((symbol, start + timedelta(days=i)) in already_saved
                           for i in range((end - start).days + 1)):
                        print(f"{symbol} {start}..{end}: using existing saved responses", flush=True)
                        continue
                    collector.collect_window(symbol, start, end)
                if index % 4 == 0:
                    export(args.end)
        finally:
            export(args.end)
    else:
        export(args.end)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate generated news outputs and report known coverage warnings."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from common import PROJECT_ROOT, load_config, parse_utc, read_jsonl


REQUIRED_FIELDS = {
    "article_id",
    "queried_tickers",
    "headline",
    "summary",
    "source",
    "url",
    "published_at_utc",
    "retrieved_at_utc",
    "relevance_type",
    "event_category",
    "relevance_reason",
    "excluded",
    "exclusion_reason",
    "duplicate_group_id",
}


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def load_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        fail(errors, f"Missing output: {path.relative_to(PROJECT_ROOT)}")
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_FIELDS - fields
        if missing:
            fail(errors, f"{path.name} is missing fields: {', '.join(sorted(missing))}")
        return list(reader)


def validate_articles(
    rows: list[dict[str, str]],
    start: date,
    end: date,
    configured_tickers: set[str],
    errors: list[str],
) -> None:
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for number, row in enumerate(rows, start=2):
        article_id = row.get("article_id", "")
        normalized_url = row.get("normalized_url", "")
        if article_id:
            if article_id in seen_ids:
                fail(errors, f"Duplicate article_id {article_id} at classified CSV line {number}")
            seen_ids.add(article_id)
        if normalized_url:
            if normalized_url in seen_urls:
                fail(errors, f"Duplicate normalized_url at classified CSV line {number}")
            seen_urls.add(normalized_url)
        published = row.get("published_at_utc", "")
        if published:
            try:
                published_date = parse_utc(published).date()
            except ValueError:
                fail(errors, f"Invalid published_at_utc at classified CSV line {number}")
            else:
                if not start <= published_date <= end:
                    fail(errors, f"Publication outside requested range at classified CSV line {number}")
            if not published.endswith("Z"):
                fail(errors, f"Publication timestamp is not serialized as UTC at line {number}")
        retrieved = row.get("retrieved_at_utc", "")
        if retrieved and not retrieved.endswith("Z"):
            fail(errors, f"Retrieval timestamp is not serialized as UTC at line {number}")
        queried = {ticker for ticker in row.get("queried_tickers", "").split("|") if ticker}
        if not queried or not queried <= configured_tickers:
            fail(errors, f"Invalid queried_tickers at classified CSV line {number}")
        if row.get("excluded") not in {"true", "false"}:
            fail(errors, f"Invalid excluded value at classified CSV line {number}")


def latest_manifest_rows() -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(PROJECT_ROOT / "raw" / "request_manifest.jsonl"):
        if row.get("record_type") == "request_window" and row.get("window_id"):
            latest[str(row["window_id"])] = row
    return list(latest.values())


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    config = load_config()
    start = date.fromisoformat(config["collection"]["start_date"])
    end = date.fromisoformat(config["collection"]["end_date"])
    tickers = set(config["collection"]["tickers"])
    all_rows = load_csv(PROJECT_ROOT / "processed" / "all_classified_articles.csv", errors)
    signals = load_csv(PROJECT_ROOT / "processed" / "signal_articles.csv", errors)
    samples = load_csv(PROJECT_ROOT / "reports" / "sample_retained_articles.csv", errors)
    validate_articles(all_rows, start, end, tickers, errors)

    canonical_all = {
        row.get("article_id") or row.get("normalized_url") or row.get("headline") for row in all_rows
    }
    for number, row in enumerate(signals, start=2):
        canonical = row.get("article_id") or row.get("normalized_url") or row.get("headline")
        if canonical not in canonical_all:
            fail(errors, f"Signal CSV line {number} is absent from the audit CSV")
        if row.get("excluded") != "false":
            fail(errors, f"Excluded article found in signal CSV line {number}")
        if row.get("relevance_type") not in {"direct", "indirect"}:
            fail(errors, f"Non-relevant article found in signal CSV line {number}")
        if row.get("event_representative") != "true":
            fail(errors, f"Non-representative event article found in signal CSV line {number}")

    sample_counts = Counter(row.get("relevance_type") for row in samples)
    if len(samples) != 20 or sample_counts != {"direct": 10, "indirect": 10}:
        fail(errors, f"Expected 10 direct and 10 indirect samples; found {dict(sample_counts)}")

    report_path = PROJECT_ROOT / "reports" / "coverage_report.json"
    if not report_path.exists():
        fail(errors, "Missing reports/coverage_report.json")
        report = {}
    else:
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    report_summary = report.get("summary", {})
    actual_signal_counts = Counter(row.get("relevance_type") for row in signals)
    if report_summary.get("retained_direct") != actual_signal_counts["direct"]:
        fail(errors, "Coverage report direct count does not match signal CSV")
    if report_summary.get("retained_indirect") != actual_signal_counts["indirect"]:
        fail(errors, "Coverage report indirect count does not match signal CSV")
    if not report.get("preflight", {}).get("coverage_verified"):
        fail(errors, "Historical endpoint access was not verified at both requested boundaries")

    for row in latest_manifest_rows():
        raw_file = row.get("raw_file")
        if raw_file and not (PROJECT_ROOT / str(raw_file)).exists():
            fail(errors, f"Manifest references missing raw file: {raw_file}")
        status = str(row.get("status") or "")
        if status.startswith("failed") or status.startswith("unavailable"):
            warnings.append(
                f"{row.get('ticker')} {row.get('from_date')}..{row.get('to_date')}: {status}"
            )
        if status == "success_empty_unverified":
            warnings.append(
                f"{row.get('ticker')} {row.get('from_date')}..{row.get('to_date')}: empty response unverified"
            )
        if status == "success_truncation_suspected":
            warnings.append(
                f"{row.get('ticker')} {row.get('from_date')}..{row.get('to_date')}: potential truncation"
            )

    if errors:
        print("OUTPUT VALIDATION FAILED", file=sys.stderr)
        for message in errors:
            print(f"ERROR: {message}", file=sys.stderr)
        return 1
    print(
        f"OUTPUT VALIDATION PASSED: {len(all_rows)} audit rows, {len(signals)} signals, "
        f"{len(samples)} samples"
    )
    if warnings:
        print(f"COVERAGE WARNINGS ({len(warnings)}):")
        for message in warnings:
            print(f"WARNING: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

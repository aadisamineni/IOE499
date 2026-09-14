#!/usr/bin/env python3
"""Resumable Finnhub company-news collector with raw-response preservation."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common import (
    PROJECT_ROOT,
    append_jsonl,
    atomic_write_json,
    load_config,
    read_jsonl,
    stable_id,
    utc_now_iso,
)


MANIFEST_PATH = PROJECT_ROOT / "raw" / "request_manifest.jsonl"
TERMINAL_SUCCESS_STATUSES = {
    "success",
    "success_empty_unverified",
    "success_truncation_suspected",
}
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}
SAFE_RESPONSE_HEADERS = {
    "content-type",
    "date",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
}


@dataclass
class ApiResponse:
    status: int | None
    body: Any
    headers: dict[str, str]
    retrieved_at_utc: str
    error_kind: str = ""


class RatePacer:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request_started: float | None = None

    def wait(self) -> None:
        if self.last_request_started is not None:
            elapsed = time.monotonic() - self.last_request_started
            remaining = self.delay_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_started = time.monotonic()


def safe_headers(headers: Any) -> dict[str, str]:
    if not headers:
        return {}
    return {
        str(key).lower(): str(value)
        for key, value in headers.items()
        if str(key).lower() in SAFE_RESPONSE_HEADERS
    }


def decode_body(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"non_json_response": text}


class FinnhubCollector:
    def __init__(self, config: dict[str, Any], api_key: str) -> None:
        self.config = config
        self.api_key = api_key
        api = config["api"]
        self.endpoint = api["base_url"].rstrip("/") + api["company_news_endpoint"]
        self.timeout = float(api["timeout_seconds"])
        self.max_retries = int(api["max_retries"])
        self.backoff_initial = float(api["backoff_initial_seconds"])
        self.backoff_max = float(api["backoff_max_seconds"])
        self.initial_window_days = int(api["initial_window_days"])
        self.minimum_window_days = int(api["minimum_window_days"])
        self.truncation_count = int(api["potential_truncation_count"])
        self.pacer = RatePacer(float(api["request_delay_seconds"]))
        self.latest_manifest = self._load_latest_manifest()

    def _load_latest_manifest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(MANIFEST_PATH):
            if row.get("record_type") == "request_window" and row.get("window_id"):
                latest[str(row["window_id"])] = row
        return latest

    def _request_once(self, ticker: str, start: date, end: date) -> ApiResponse:
        # Finnhub's documented authentication mechanism is the token query parameter.
        # The constructed URL is deliberately never logged, stored, or included in errors.
        query = urlencode(
            {
                "symbol": ticker,
                "from": start.isoformat(),
                "to": end.isoformat(),
                "token": self.api_key,
            }
        )
        request = Request(
            f"{self.endpoint}?{query}",
            headers={"Accept": "application/json", "User-Agent": "nvda-news-research/1.0"},
        )
        self.pacer.wait()
        retrieved_at = utc_now_iso()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return ApiResponse(
                    status=response.status,
                    body=decode_body(response.read()),
                    headers=safe_headers(response.headers),
                    retrieved_at_utc=retrieved_at,
                )
        except HTTPError as exc:
            return ApiResponse(
                status=exc.code,
                body=decode_body(exc.read()),
                headers=safe_headers(exc.headers),
                retrieved_at_utc=retrieved_at,
                error_kind="http_error",
            )
        except (URLError, TimeoutError, socket.timeout):
            # Exception strings may contain the authenticated URL, so retain only the type.
            return ApiResponse(
                status=None,
                body=None,
                headers={},
                retrieved_at_utc=retrieved_at,
                error_kind="network_or_timeout",
            )

    @staticmethod
    def _window_id(ticker: str, start: date, end: date) -> str:
        return stable_id("window", [ticker, start.isoformat(), end.isoformat()])

    @staticmethod
    def _relative(path: Path) -> str:
        return str(path.relative_to(PROJECT_ROOT))

    def _raw_path(self, ticker: str, start: date, end: date) -> Path:
        return (
            PROJECT_ROOT
            / "raw"
            / "responses"
            / ticker
            / f"{start.isoformat()}_{end.isoformat()}.json"
        )

    def _error_path(self, ticker: str, start: date, end: date, attempt: int) -> Path:
        return (
            PROJECT_ROOT
            / "raw"
            / "errors"
            / ticker
            / f"{start.isoformat()}_{end.isoformat()}_attempt-{attempt}.json"
        )

    def _record_manifest(self, row: dict[str, Any]) -> None:
        append_jsonl(MANIFEST_PATH, row)
        self.latest_manifest[str(row["window_id"])] = row

    def _base_manifest_row(self, ticker: str, start: date, end: date) -> dict[str, Any]:
        return {
            "record_type": "request_window",
            "window_id": self._window_id(ticker, start, end),
            "ticker": ticker,
            "from_date": start.isoformat(),
            "to_date": end.isoformat(),
        }

    def _write_error_response(
        self,
        ticker: str,
        start: date,
        end: date,
        attempt: int,
        response: ApiResponse,
    ) -> Path:
        path = self._error_path(ticker, start, end, attempt)
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "request": {
                    "endpoint": "/company-news",
                    "symbol": ticker,
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                },
                "retrieved_at_utc": response.retrieved_at_utc,
                "http_status": response.status,
                "response_headers": response.headers,
                "error_kind": response.error_kind,
                "response": response.body,
            },
        )
        return path

    def fetch_with_retries(self, ticker: str, start: date, end: date) -> ApiResponse:
        backoff = self.backoff_initial
        base = self._base_manifest_row(ticker, start, end)
        for attempt in range(1, self.max_retries + 2):
            response = self._request_once(ticker, start, end)
            if response.status == 200:
                return response

            error_path = self._write_error_response(ticker, start, end, attempt, response)
            is_transient = (
                response.status in TRANSIENT_HTTP_STATUSES
                or response.status is None
                or (response.status is not None and 500 <= response.status <= 599)
            )
            if is_transient and attempt <= self.max_retries:
                retry_after = response.headers.get("retry-after", "")
                try:
                    wait_seconds = max(backoff, float(retry_after))
                except ValueError:
                    wait_seconds = backoff
                wait_seconds = min(wait_seconds, self.backoff_max)
                self._record_manifest(
                    {
                        **base,
                        "status": "transient_retry",
                        "attempt": attempt,
                        "http_status": response.status,
                        "retrieved_at_utc": response.retrieved_at_utc,
                        "error_kind": response.error_kind,
                        "raw_error_file": self._relative(error_path),
                        "retry_wait_seconds": wait_seconds,
                    }
                )
                print(
                    f"{ticker} {start}..{end}: transient failure "
                    f"(status={response.status}); retry {attempt}/{self.max_retries}"
                )
                time.sleep(wait_seconds)
                backoff = min(backoff * 2, self.backoff_max)
                continue
            return response
        raise AssertionError("retry loop exited unexpectedly")

    def _status_for_failure(self, response: ApiResponse) -> str:
        if response.status in {401, 403}:
            return "unavailable_auth_or_plan"
        if response.status == 429:
            return "failed_rate_limit"
        if response.status is None:
            return "failed_network_or_timeout"
        if response.status and response.status >= 500:
            return "failed_server"
        return "failed_http"

    def collect_window(self, ticker: str, start: date, end: date) -> None:
        window_id = self._window_id(ticker, start, end)
        prior = self.latest_manifest.get(window_id)
        if prior:
            prior_status = prior.get("status")
            if prior_status in TERMINAL_SUCCESS_STATUSES:
                raw_file = prior.get("raw_file")
                if raw_file and (PROJECT_ROOT / str(raw_file)).exists():
                    prior_count = int(prior.get("article_count") or 0)
                    span_days = (end - start).days + 1
                    if prior_count >= self.truncation_count and span_days > self.minimum_window_days:
                        revised = {
                            **prior,
                            "status": "split_potential_truncation",
                            "reclassified_at_utc": utc_now_iso(),
                            "reclassification_reason": (
                                "Saved count meets the current conservative truncation threshold."
                            ),
                        }
                        self._record_manifest(revised)
                        print(
                            f"{ticker} {start}..{end}: resume reclassify and split "
                            f"({prior_count} >= {self.truncation_count})"
                        )
                        left, right = split_window(start, end)
                        self.collect_window(ticker, *left)
                        self.collect_window(ticker, *right)
                        return
                    if prior_count >= self.truncation_count:
                        revised = {
                            **prior,
                            "status": "success_truncation_suspected",
                            "reclassified_at_utc": utc_now_iso(),
                            "reclassification_reason": (
                                "One-day saved count meets the current conservative truncation threshold."
                            ),
                        }
                        self._record_manifest(revised)
                        print(
                            f"{ticker} {start}..{end}: resume flag potential truncation "
                            f"({prior_count} >= {self.truncation_count})"
                        )
                        return
                    print(f"{ticker} {start}..{end}: resume skip ({prior_status})")
                    return
            if prior_status == "split_potential_truncation":
                left, right = split_window(start, end)
                self.collect_window(ticker, *left)
                self.collect_window(ticker, *right)
                return

        response = self.fetch_with_retries(ticker, start, end)
        base = self._base_manifest_row(ticker, start, end)
        if response.status != 200:
            status = self._status_for_failure(response)
            row = {
                **base,
                "status": status,
                "attempt": self.max_retries + 1,
                "http_status": response.status,
                "retrieved_at_utc": response.retrieved_at_utc,
                "error_kind": response.error_kind,
                "article_count": None,
            }
            self._record_manifest(row)
            print(f"{ticker} {start}..{end}: {status} (status={response.status})")
            if response.status == 401:
                raise RuntimeError("Finnhub rejected FINNHUB_API_KEY (HTTP 401)")
            return

        if not isinstance(response.body, list):
            row = {
                **base,
                "status": "unavailable_invalid_response",
                "http_status": response.status,
                "retrieved_at_utc": response.retrieved_at_utc,
                "article_count": None,
            }
            self._record_manifest(row)
            print(f"{ticker} {start}..{end}: unavailable_invalid_response")
            return

        article_count = len(response.body)
        span_days = (end - start).days + 1
        should_split = (
            article_count >= self.truncation_count and span_days > self.minimum_window_days
        )
        if should_split:
            status = "split_potential_truncation"
            terminal = False
        elif article_count >= self.truncation_count:
            status = "success_truncation_suspected"
            terminal = True
        elif article_count == 0:
            status = "success_empty_unverified"
            terminal = True
        else:
            status = "success"
            terminal = True

        raw_path = self._raw_path(ticker, start, end)
        atomic_write_json(
            raw_path,
            {
                "schema_version": 1,
                "request": {
                    "endpoint": "/company-news",
                    "symbol": ticker,
                    "from": start.isoformat(),
                    "to": end.isoformat(),
                },
                "retrieved_at_utc": response.retrieved_at_utc,
                "http_status": response.status,
                "response_headers": response.headers,
                "collection_status": status,
                "terminal_window": terminal,
                "response": response.body,
            },
        )
        self._record_manifest(
            {
                **base,
                "status": status,
                "http_status": response.status,
                "retrieved_at_utc": response.retrieved_at_utc,
                "article_count": article_count,
                "raw_file": self._relative(raw_path),
            }
        )
        print(f"{ticker} {start}..{end}: {status}, {article_count} articles")

        if should_split:
            left, right = split_window(start, end)
            self.collect_window(ticker, *left)
            self.collect_window(ticker, *right)

    def collect(self, tickers: list[str]) -> None:
        collection = self.config["collection"]
        start = date.fromisoformat(collection["start_date"])
        end = date.fromisoformat(collection["end_date"])
        for ticker in tickers:
            for window_start, window_end in date_windows(start, end, self.initial_window_days):
                self.collect_window(ticker, window_start, window_end)

    def preflight(self) -> dict[str, Any]:
        collection = self.config["collection"]
        ticker = collection["primary_ticker"]
        start = date.fromisoformat(collection["start_date"])
        end = date.fromisoformat(collection["end_date"])
        probes = [
            ("oldest_requested_dates", start, min(start + timedelta(days=1), end)),
            ("newest_requested_dates", max(start, end - timedelta(days=1)), end),
        ]
        results: list[dict[str, Any]] = []
        for label, probe_start, probe_end in probes:
            response = self.fetch_with_retries(ticker, probe_start, probe_end)
            body_is_list = isinstance(response.body, list)
            count = len(response.body) if body_is_list else None
            status = (
                "confirmed_nonempty"
                if response.status == 200 and body_is_list and count and count > 0
                else "empty_unverified"
                if response.status == 200 and body_is_list
                else "unavailable"
            )
            raw_path = (
                PROJECT_ROOT
                / "raw"
                / "preflight"
                / f"{ticker}_{probe_start.isoformat()}_{probe_end.isoformat()}.json"
            )
            atomic_write_json(
                raw_path,
                {
                    "schema_version": 1,
                    "request": {
                        "endpoint": "/company-news",
                        "symbol": ticker,
                        "from": probe_start.isoformat(),
                        "to": probe_end.isoformat(),
                    },
                    "retrieved_at_utc": response.retrieved_at_utc,
                    "http_status": response.status,
                    "response_headers": response.headers,
                    "preflight_status": status,
                    "response": response.body,
                },
            )
            results.append(
                {
                    "label": label,
                    "ticker": ticker,
                    "from_date": probe_start.isoformat(),
                    "to_date": probe_end.isoformat(),
                    "http_status": response.status,
                    "article_count": count,
                    "status": status,
                    "retrieved_at_utc": response.retrieved_at_utc,
                    "raw_file": self._relative(raw_path),
                }
            )
            print(
                f"preflight {label}: {status}, status={response.status}, articles={count}"
            )

        report = {
            "generated_at_utc": utc_now_iso(),
            "requested_start_date": start.isoformat(),
            "requested_end_date": end.isoformat(),
            "coverage_verified": all(row["status"] == "confirmed_nonempty" for row in results),
            "verification_method": (
                "Successful non-empty NVDA responses at both ends of the requested range. "
                "This verifies endpoint access, not source completeness."
            ),
            "probes": results,
        }
        atomic_write_json(PROJECT_ROOT / "raw" / "preflight_report.json", report)
        return report


def date_windows(start: date, end: date, days: int):
    cursor = start
    while cursor <= end:
        window_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, window_end
        cursor = window_end + timedelta(days=1)


def split_window(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    if start >= end:
        raise ValueError("Cannot split a one-day window")
    midpoint = start + timedelta(days=(end - start).days // 2)
    return (start, midpoint), (midpoint + timedelta(days=1), end)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to config JSON (defaults to project config.json)")
    parser.add_argument(
        "--preflight-only", action="store_true", help="Verify historical access without full collection"
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", help="Skip the endpoint-access probes"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Optional subset of configured tickers, useful for controlled retries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        print("FINNHUB_API_KEY is required but was not found in the environment.", file=sys.stderr)
        return 2

    config = load_config(args.config)
    configured_tickers = config["collection"]["tickers"]
    tickers = [ticker.upper() for ticker in (args.tickers or configured_tickers)]
    unknown = sorted(set(tickers) - set(configured_tickers))
    if unknown:
        print(f"Tickers are not in config.json: {', '.join(unknown)}", file=sys.stderr)
        return 2

    collector = FinnhubCollector(config, api_key)
    if not args.skip_preflight:
        report = collector.preflight()
        if not report["coverage_verified"]:
            print(
                "Historical access was not fully verified; inspect raw/preflight_report.json. "
                "Collection was not started.",
                file=sys.stderr,
            )
            return 3
    if not args.preflight_only:
        collector.collect(tickers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

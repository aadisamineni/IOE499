"""Resumably download the highest-resolution NVDA data available from Alpaca.

The default run covers 2025-10-01 through 2026-05-01 inclusive and downloads
tick trades, tick quotes, one-minute bars, auction records, and corporate
actions. Credentials are read without logging from ../Alpaca_api_key/ or from
the standard APCA environment variables.

Run from the repository root:
    .venv/bin/python "alpaca data pull/pull_alpaca_nvda.py"
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "raw"
STATE_ROOT = HERE / ".state"
METADATA = HERE / "metadata"
MANIFEST = METADATA / "request_manifest.jsonl"
SUMMARY = METADATA / "summary.json"
RUN_CONFIG = METADATA / "run_config.json"
PUBLIC_KEY_FILE = ROOT / "Alpaca_api_key/Public_key.end"
PRIVATE_KEY_FILE = ROOT / "Alpaca_api_key/Private_key.env"
BASE_URL = "https://data.alpaca.markets"
SCRIPT_VERSION = 1
MANIFEST_LOCK = threading.Lock()

KIND_CONFIG = {
    "bars": {
        "path": "/v2/stocks/{symbol}/bars",
        "response_key": "bars",
        "params": {"timeframe": "1Min", "adjustment": "raw"},
        "feeds": True,
        "daily": True,
    },
    "trades": {
        "path": "/v2/stocks/{symbol}/trades",
        "response_key": "trades",
        "params": {},
        "feeds": True,
        "daily": True,
    },
    "quotes": {
        "path": "/v2/stocks/{symbol}/quotes",
        "response_key": "quotes",
        "params": {},
        "feeds": True,
        "daily": True,
    },
    "auctions": {
        "path": "/v2/stocks/auctions",
        "response_key": "auctions",
        "params": {},
        "feeds": False,
        "daily": True,
    },
    "corporate_actions": {
        "path": "/v1/corporate-actions",
        "response_key": "corporate_actions",
        "params": {"data_quality": "all", "region": "us"},
        "limit": 1000,
        "feeds": False,
        "daily": False,
    },
}


class ApiError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Alpaca HTTP {status_code}: {message}")
        self.status_code = status_code
        self.message = message


def normalize_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def read_labeled_secret(path: Path, accepted_labels: set[str]) -> str:
    """Read one raw token or label=value entry without exposing its value."""
    if not path.is_file():
        raise FileNotFoundError(f"Credential file is missing: {path}")
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            label, value = line.split("=", 1)
            if normalize_name(label) in accepted_labels:
                entries.append(value.strip().strip("'\""))
        else:
            entries.append(line.strip("'\""))
    entries = [value for value in entries if value]
    if len(entries) != 1:
        raise ValueError(f"Expected exactly one usable credential in {path}")
    return entries[0]


def load_credentials() -> tuple[str, str]:
    key_id = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not key_id:
        key_id = read_labeled_secret(
            PUBLIC_KEY_FILE, {"publickey", "apikeyid", "apcaapikeyid"})
    if not secret:
        secret = read_labeled_secret(
            PRIVATE_KEY_FILE, {"privatekey", "secretkey", "apcaapisecretkey"})
    return key_id, secret


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def daily_intervals(start: date, end: date) -> Iterable[tuple[date, str, str]]:
    current = start
    while current <= end:
        lower = datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc)
        upper = f"{current.isoformat()}T23:59:59.999999999Z"
        yield current, iso_utc(lower), upper
        current += timedelta(days=1)


def message_from_response(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500].replace("\n", " ")
    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("error") or payload)[:500]
    return str(payload)[:500]


class RequestRateLimiter:
    """Space requests across threads to stay below an account-wide RPM cap."""

    def __init__(self, requests_per_minute: int):
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be positive")
        self.minimum_interval = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_start = 0.0
        self.requests = 0

    def acquire(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_start - now)
            self.next_start = max(now, self.next_start) + self.minimum_interval
            self.requests += 1
        if delay:
            time.sleep(delay)


class AlpacaClient:
    def __init__(self, key_id: str, secret: str, timeout: int = 60, retries: int = 8,
                 rate_limiter: RequestRateLimiter | None = None):
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret,
            "Accept": "application/json",
            "User-Agent": "IOE499-NVDA-research/1.0",
        })
        self.timeout = timeout
        self.retries = retries
        self.rate_limiter = rate_limiter
        self.request_count = 0

    def get(self, path: str, params: dict[str, Any], allow_status: set[int] | None = None) -> requests.Response:
        allowed = allow_status or set()
        for attempt in range(self.retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire()
            try:
                response = self.session.get(
                    BASE_URL + path, params=params, timeout=self.timeout)
            except requests.RequestException as error:
                if attempt >= self.retries:
                    raise RuntimeError(f"Alpaca request failed after retries: {error}") from error
                delay = min(60.0, 2 ** attempt + random.random())
                print(f"network retry {attempt + 1}/{self.retries} in {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            self.request_count += 1
            if response.status_code == 200 or response.status_code in allowed:
                return response
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt >= self.retries:
                    raise ApiError(response.status_code, message_from_response(response))
                retry_after = response.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = min(60.0, 2 ** attempt + random.random())
                delay = min(60.0, max(0.5, delay))
                print(
                    f"HTTP {response.status_code} retry {attempt + 1}/{self.retries} "
                    f"in {delay:.1f}s", flush=True)
                time.sleep(delay)
                continue
            raise ApiError(response.status_code, message_from_response(response))
        raise AssertionError("unreachable")


def normalize_records(payload: Any, symbol: str) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [record for record in payload if isinstance(record, dict)]
    if isinstance(payload, dict):
        symbol_records = payload.get(symbol)
        if isinstance(symbol_records, list):
            return [record for record in symbol_records if isinstance(record, dict)]
        # Corporate-action responses may group arrays by action type.
        collected = []
        for action_type, values in payload.items():
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, dict):
                        collected.append({"_action_type": action_type, **value})
        return collected
    raise ValueError(f"Unexpected Alpaca response payload type: {type(payload).__name__}")


def response_records(data: dict[str, Any], response_key: str, symbol: str) -> list[dict[str, Any]]:
    payload = data.get(response_key)
    if payload is None and response_key == "corporate_actions":
        payload = data.get("corporateActions") or data.get("actions")
    return normalize_records(payload, symbol)


def query_fingerprint(kind: str, symbol: str, feed: str | None,
                      start: str, end: str, extra: dict[str, Any]) -> str:
    canonical = json.dumps({
        "version": SCRIPT_VERSION,
        "kind": kind,
        "symbol": symbol,
        "feed": feed,
        "start": start,
        "end": end,
        "extra": extra,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl_gzip(path: Path, records: list[dict[str, Any]],
                     symbol: str, feed: str | None, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                for record in records:
                    enriched = {"_symbol": symbol, "_kind": kind, **record}
                    if feed:
                        enriched = {"_feed": feed, **enriched}
                    text.write(json.dumps(
                        enriched, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest_value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest_value.update(block)
    return digest_value.hexdigest()


def append_manifest(entry: dict[str, Any]) -> None:
    with MANIFEST_LOCK:
        METADATA.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            destination.flush()
            os.fsync(destination.fileno())


def load_manifest() -> dict[str, dict[str, Any]]:
    latest = {}
    with MANIFEST_LOCK:
        if not MANIFEST.is_file():
            return latest
        lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if line.strip():
            entry = json.loads(line)
            latest[entry["relative_path"]] = entry
    return latest


def combine_chunks(chunks: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    os.replace(temporary, destination)


def download_interval(
        client: AlpacaClient,
        output_root: Path,
        kind: str,
        symbol: str,
        feed: str | None,
        interval_label: str,
        start: str,
        end: str,
        manifest: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config = KIND_CONFIG[kind]
    feed_label = feed or ("reference" if kind == "corporate_actions" else "sip")
    destination = output_root / feed_label / kind / f"{symbol}_{kind}_{interval_label}.jsonl.gz"
    relative = str(destination.relative_to(ROOT))
    prior = manifest.get(relative)
    if destination.is_file() and prior and prior.get("status") == "complete":
        print(f"skip complete {feed_label}/{kind}/{interval_label} rows={prior['rows']}", flush=True)
        return prior

    extra = dict(config["params"])
    fingerprint = query_fingerprint(kind, symbol, feed, start, end, extra)
    state_dir = STATE_ROOT / feed_label / kind / interval_label
    state_path = state_dir / "state.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise ValueError(f"Existing resume state does not match query: {state_dir}")
    else:
        state = {
            "fingerprint": fingerprint,
            "next_page_token": None,
            "pages": 0,
            "rows": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
        atomic_json(state_path, state)

    expected_chunks = state["pages"]
    for orphan in sorted(state_dir.glob("page_*.jsonl.gz"))[expected_chunks:]:
        orphan.unlink()

    while True:
        params: dict[str, Any] = {
            "start": start,
            "end": end,
            "limit": config.get("limit", 10000),
            "sort": "asc",
            **extra,
        }
        if kind != "corporate_actions":
            params["currency"] = "USD"
        if config["feeds"]:
            params["feed"] = feed
        if kind in {"auctions", "corporate_actions"}:
            params["symbols"] = symbol
        if state["next_page_token"]:
            params["page_token"] = state["next_page_token"]
        path = config["path"].format(symbol=symbol)
        response = client.get(path, params)
        data = response.json()
        records = response_records(data, config["response_key"], symbol)
        page = int(state["pages"])
        chunk = state_dir / f"page_{page:06d}.jsonl.gz"
        write_jsonl_gzip(chunk, records, symbol, feed, kind)

        timestamps = [
            str(record.get("t") or record.get("timestamp") or record.get("process_date") or "")
            for record in records
        ]
        timestamps = [value for value in timestamps if value]
        if timestamps and state["first_timestamp"] is None:
            state["first_timestamp"] = timestamps[0]
        if timestamps:
            state["last_timestamp"] = timestamps[-1]
        state["pages"] = page + 1
        state["rows"] = int(state["rows"]) + len(records)
        state["next_page_token"] = data.get("next_page_token")
        atomic_json(state_path, state)
        # Tick datasets can require hundreds of pages per day. Keep progress
        # visible without flooding long-running terminals with one line per page.
        if page == 0 or (page + 1) % 25 == 0 or not state["next_page_token"]:
            print(
                f"page {feed_label}/{kind}/{interval_label} #{page + 1} "
                f"+{len(records)} total={state['rows']}", flush=True)
        if not state["next_page_token"]:
            break

    chunks = sorted(state_dir.glob("page_*.jsonl.gz"))
    if len(chunks) != state["pages"]:
        raise AssertionError(f"Resume chunk count mismatch for {state_dir}")
    combine_chunks(chunks, destination)
    entry = {
        "status": "complete",
        "kind": kind,
        "symbol": symbol,
        "feed": feed_label,
        "interval": interval_label,
        "start_utc": start,
        "end_utc": end,
        "rows": int(state["rows"]),
        "pages": int(state["pages"]),
        "first_timestamp": state["first_timestamp"],
        "last_timestamp": state["last_timestamp"],
        "compressed_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "relative_path": relative,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_fingerprint": fingerprint,
    }
    append_manifest(entry)
    with MANIFEST_LOCK:
        manifest[relative] = entry
    shutil.rmtree(state_dir)
    print(
        f"complete {feed_label}/{kind}/{interval_label} rows={entry['rows']} "
        f"size={entry['compressed_bytes']}", flush=True)
    return entry


def feed_available(client: AlpacaClient, feed: str, symbol: str) -> tuple[bool, str]:
    response = client.get(
        f"/v2/stocks/{symbol}/bars",
        {
            "feed": feed,
            "timeframe": "1Min",
            "adjustment": "raw",
            "start": "2026-03-02T14:30:00Z",
            "end": "2026-03-02T14:31:00Z",
            "limit": 1,
        },
        allow_status={400, 403, 422},
    )
    if response.status_code == 200:
        return True, "available"
    return False, message_from_response(response)


def choose_feeds(client: AlpacaClient, requested: list[str], symbol: str,
                 include_overnight: bool) -> tuple[list[str], dict[str, str]]:
    if requested != ["auto"]:
        candidates = requested
    else:
        candidates = ["sip", "iex"]
    selected = []
    audit = {}
    for feed in candidates:
        available, reason = feed_available(client, feed, symbol)
        audit[feed] = reason
        if available:
            selected.append(feed)
            if requested == ["auto"] and feed == "sip":
                break
    if not selected:
        raise RuntimeError(f"No requested stock feed is available: {audit}")
    if include_overnight and "boats" not in selected:
        available, reason = feed_available(client, "boats", symbol)
        audit["boats"] = reason
        if available:
            selected.append("boats")
    return selected, audit


def update_summary(run_config: dict[str, Any]) -> dict[str, Any]:
    manifest = load_manifest()
    entries = list(manifest.values())
    by_feed_kind: dict[str, dict[str, dict[str, int]]] = {}
    for entry in entries:
        feed = entry["feed"]
        kind = entry["kind"]
        aggregate = by_feed_kind.setdefault(feed, {}).setdefault(
            kind, {"files": 0, "rows": 0, "compressed_bytes": 0})
        aggregate["files"] += 1
        aggregate["rows"] += int(entry["rows"])
        aggregate["compressed_bytes"] += int(entry["compressed_bytes"])
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if run_config.get("complete") else "in_progress",
        "symbol": run_config["symbol"],
        "start_inclusive": run_config["start_inclusive"],
        "end_inclusive": run_config["end_inclusive"],
        "selected_feeds": run_config["selected_feeds"],
        "feed_probe": run_config["feed_probe"],
        "credentials_logged": False,
        "file_format": "gzip-compressed newline-delimited JSON; Alpaca fields preserved",
        "manifest_entries": len(entries),
        "by_feed_and_kind": by_feed_kind,
        "api_requests_this_run": run_config.get("api_requests_this_run"),
        "documentation": {
            "bars": "https://docs.alpaca.markets/us/reference/stockbarsingle-1",
            "trades": "https://docs.alpaca.markets/us/v1.4.2/reference/stocktradesingle-1",
            "quotes": "https://docs.alpaca.markets/us/reference/stockquotes-1",
            "auctions": "https://docs.alpaca.markets/us/reference/stockauctions-1",
            "corporate_actions": "https://docs.alpaca.markets/us/reference/corporateactions-1",
        },
    }
    atomic_json(SUMMARY, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2025, 10, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 5, 1))
    parser.add_argument(
        "--kinds", nargs="+", choices=sorted(KIND_CONFIG),
        default=["bars", "trades", "quotes", "auctions", "corporate_actions"])
    parser.add_argument(
        "--feeds", nargs="+", choices=["auto", "sip", "iex", "boats"], default=["auto"])
    parser.add_argument("--no-overnight", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-days", type=int, help="Limit daily intervals for a test run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=int, default=190)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")
    if args.max_days is not None and args.max_days < 1:
        raise SystemExit("--max-days must be positive")
    if args.workers < 1 or args.requests_per_minute < 1:
        raise SystemExit("--workers and --requests-per-minute must be positive")
    symbol = args.symbol.upper().strip()
    output_root = args.output if args.output.is_absolute() else ROOT / args.output
    key_id, secret = load_credentials()
    rate_limiter = RequestRateLimiter(args.requests_per_minute)
    client = AlpacaClient(key_id, secret, rate_limiter=rate_limiter)
    feeds, feed_audit = choose_feeds(
        client, args.feeds, symbol, include_overnight=not args.no_overnight)
    print(f"authenticated; selected feeds={','.join(feeds)}", flush=True)
    for feed, result in feed_audit.items():
        print(f"feed probe {feed}: {result}", flush=True)

    config = {
        "script_version": SCRIPT_VERSION,
        "symbol": symbol,
        "start_inclusive": str(args.start),
        "end_inclusive": str(args.end),
        "kinds": args.kinds,
        "selected_feeds": feeds,
        "feed_probe": feed_audit,
        "raw_output": str(output_root.relative_to(ROOT)),
        "credentials_source": (
            "APCA environment variables when set; otherwise local ignored Alpaca_api_key files"
        ),
        "credentials_logged": False,
        "daily_partition_timezone": "UTC",
        "end_policy": "daily upper bound is 23:59:59.999999999 UTC",
        "workers": args.workers,
        "requests_per_minute_cap": args.requests_per_minute,
        "complete": False,
    }
    atomic_json(RUN_CONFIG, config)
    if args.probe_only:
        config["api_requests_this_run"] = rate_limiter.requests
        update_summary(config)
        return

    manifest = load_manifest()
    intervals = list(daily_intervals(args.start, args.end))
    if args.max_days is not None:
        intervals = intervals[:args.max_days]

    worker_local = threading.local()

    def worker_client() -> AlpacaClient:
        if not hasattr(worker_local, "client"):
            worker_local.client = AlpacaClient(
                key_id, secret, rate_limiter=rate_limiter)
        return worker_local.client

    def execute(task: tuple[str, str | None, str, str, str]) -> dict[str, Any]:
        kind, feed, label, start, end = task
        return download_interval(
            worker_client(), output_root, kind, symbol, feed,
            label, start, end, manifest)

    try:
        for kind in args.kinds:
            kind_config = KIND_CONFIG[kind]
            kind_feeds: list[str | None]
            if kind_config["feeds"]:
                kind_feeds = feeds
            else:
                kind_feeds = [None]
            if kind_config["daily"]:
                work = [(str(day), start, end) for day, start, end in intervals]
            else:
                # The corporate-actions endpoint accepts dates rather than timestamps.
                lower = str(args.start)
                upper = str(args.end)
                work = [(f"{args.start}_{args.end}", lower, upper)]
            tasks = [
                (kind, feed, label, start, end)
                for feed in kind_feeds
                if kind != "auctions" or feed in {None, "sip"}
                for label, start, end in work
            ]
            with ThreadPoolExecutor(max_workers=min(args.workers, len(tasks))) as executor:
                futures = [executor.submit(execute, task) for task in tasks]
                for future in as_completed(futures):
                    future.result()
                    config["api_requests_this_run"] = rate_limiter.requests
                    update_summary(config)
    except KeyboardInterrupt:
        config["api_requests_this_run"] = rate_limiter.requests
        update_summary(config)
        print("Interrupted; completed pages and files are resumable.", file=sys.stderr)
        raise SystemExit(130)

    config["complete"] = args.max_days is None
    config["api_requests_this_run"] = rate_limiter.requests
    atomic_json(RUN_CONFIG, config)
    summary = update_summary(config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
